"""
Strategy: Non-linear Structural Search with Linear Distillation
1. On first call, use scipy.optimize.differential_evolution to search for 
   a function structure that fits the data well. 
   The search space is a parameterized family: 
   y = a*sin(b*x + c) + d*cos(e*y + f) + g*x*h*y + i*x + j
   (Note: using 10 params to match the evaluator's param length, though 
   the internal fit might use all 10 to define a richer function if needed,
   but here we stick to a flexible mix of trig and poly terms).
   
   Actually, to be more robust, we will search for a function of the form:
   y = sum_{i=1}^{5} [ a_i * sin(b_i * x + c_i) + d_i * cos(e_i * y + f_i) ] 
   This is 30 params, too many for DE in 150s? 
   Let's use a simpler, more powerful basis:
   y = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*xy + a6*sin(x) + a7*cos(y) + a8*sin(x+y) + a9*cos(x-y)
   This is a linear basis! The current program already tries this.
   
   The key difference: The current program assumes the target is linear in some fixed basis.
   If the target is e.g. sin(x^2), a linear basis of sin(x), cos(x) cannot represent it well.
   
   New Strategy: 
   Use Differential Evolution to fit a NON-LINEAR model directly.
   Model: y = a * exp(-b*x^2 - c*y^2) * sin(d*x + e*y) + f*x + g*y + h*x^2 + i*y^2 + j
   This covers Gaussian-modulated waves plus polynomials.
   
   Once DE finds the best (a..j), we have a function f(x,y).
   We then project this f(x,y) onto a standard orthogonal basis (Chebyshev) 
   using least squares on the training points.
   We return the Chebyshev projection.
   
   Why? 
   1. DE can find the non-linear structure (e.g., the frequency d, e, or decay b, c).
   2. The projection onto Chebyshev ensures the final output is a smooth, well-conditioned 
      linear combination that the evaluator's BFGS can handle trivially (it's already at the minimum 
      for that basis, or close to it).
   3. This avoids the "local minima" trap of BFGS on a poorly chosen basis.
"""

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

_state = {
    "initialized": False,
    "best_params": None,
    "cheb_coeffs": None,
    "x_train": None,
    "y_train": None
}

def _model(x, y, params):
    """
    Non-linear model for structural search.
    params: [a, b, c, d, e, f, g, h, i, j]
    y = a * exp(-b*x^2 - c*y^2) * sin(d*x + e*y) + f*x + g*y + h*x^2 + i*y^2 + j
    """
    a, b, c, d, e, f, g, h, i, j = params
    
    # Ensure b, c are positive to avoid explosion in exp
    b_pos = np.abs(b) + 1e-6
    c_pos = np.abs(c) + 1e-6
    
    # Compute the non-linear part
    # Clip the exponent to avoid overflow/underflow issues in DE
    exponent = -b_pos * x**2 - c_pos * y**2
    exponent = np.clip(exponent, -50, 50)
    
    wave = np.sin(d * x + e * y)
    nonlinear_part = a * np.exp(exponent) * wave
    
    # Polynomial part
    poly_part = f * x + g * y + h * x**2 + i * y**2 + j
    
    return nonlinear_part + poly_part

def _objective(params, x, y):
    """
    Objective function for Differential Evolution.
    Returns MSE.
    """
    try:
        pred = _model(x, y, params)
        if not np.all(np.isfinite(pred)):
            return 1e10
        mse = np.mean((pred - y) ** 2)
        return mse
    except Exception:
        return 1e10

def _build_chebyshev_basis(x, y):
    """
    Builds a Chebyshev basis up to degree 4 (10 terms).
    T0=1, T1=x, T2=2x^2-1, T3=4x^3-3x, T4=8x^4-8x^2+1
    Same for y.
    We need a 2D basis.
    Let's use a tensor product or a specific set of 10 terms.
    Terms:
    1, x, y, x^2, y^2, xy, x^3, y^3, x^2y, xy^2
    Wait, Chebyshev is better for conditioning.
    Let's use:
    1, T1(x), T1(y), T2(x), T2(y), T1(x)T1(y), T3(x), T3(y), T2(x)T1(y), T1(x)T2(y)
    This gives 10 well-conditioned features.
    """
    T0 = np.ones_like(x)
    T1x = x
    T1y = y
    T2x = 2 * x**2 - 1
    T2y = 2 * y**2 - 1
    T3x = 4 * x**3 - 3 * x
    T3y = 4 * y**3 - 3 * y
    
    basis = np.column_stack([
        T0, T1x, T1y, T2x, T2y, T1x * T1y, T3x, T3y, T2x * T1y, T1x * T2y
    ])
    return basis

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    global _state
    
    start_time = time.time()
    deadline = start_time + 140 # 10s margin
    
    if not _state["initialized"]:
        _state["x_train"] = x
        _state["y_train"] = y
        
        # 1. Structural Search with Differential Evolution
        # Bounds for parameters
        # a: amplitude (10), b,c: decay (10), d,e: freq (10), f-j: poly coeffs (10)
        bounds = [
            (-10, 10),  # a
            (0, 10),    # b (positive)
            (0, 10),    # c (positive)
            (-10, 10),  # d
            (-10, 10),  # e
            (-10, 10),  # f
            (-10, 10),  # g
            (-10, 10),  # h
            (-10, 10),  # i
            (-10, 10)   # j
        ]
        
        best_params = None
        best_mse = float('inf')
        
        try:
            # Differential Evolution is global but slow. 
            # maxiter=100, tol=1e-6, popsize=15
            # We have 140s. DE with 10 vars, popsize 15, 100 iter = 1500 evals.
            # Each eval is fast (200 points).
            result = differential_evolution(
                _objective, bounds, args=(x, y),
                maxiter=200, tol=1e-5, popsize=20,
                seed=42, polish=True, init='sobol'
            )
            
            if result.success or result.fun < best_mse:
                best_mse = result.fun
                best_params = result.x
        except Exception:
            pass
        
        if best_params is None:
            # Fallback to zeros or ones if DE fails completely
            best_params = np.zeros(10)
            
        _state["best_params"] = best_params
        
        # 2. Distill to Linear Basis (Chebyshev)
        # We want to find coeffs c such that:
        # X_cheb @ c approx _model(x, y, best_params)
        # This is a linear least squares problem.
        
        X_cheb = _build_chebyshev_basis(x, y)
        
        # Check for finite
        if not np.all(np.isfinite(X_cheb)):
            X_cheb = np.column_stack([
                np.ones(len(x)), x, y, x**2, y**2, x*y, x**3, y**3, x**2*y, x*y**2
            ])
            
        # Get the target values from the non-linear model
        target_model = _model(x, y, best_params)
        
        # Solve least squares: X_cheb @ c = target_model
        try:
            # Use lstsq for stability
            coeffs, residuals, rank, sv = np.linalg.lstsq(X_cheb, target_model, rcond=None)
            _state["cheb_coeffs"] = coeffs
        except Exception:
            # Fallback to simple linear fit on raw data
            X_simple = np.column_stack([np.ones(len(x)), x, y, x**2, y**2, x*y, x**3, y**3, x**2*y, x*y**2])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X_simple, y, rcond=None)
                _state["cheb_coeffs"] = coeffs
            except:
                _state["cheb_coeffs"] = np.zeros(10)
                
        _state["initialized"] = True
        
        # If we ran out of time, return a valid prediction
        if time.time() > deadline:
            X_curr = _build_chebyshev_basis(x, y)
            pred = X_curr @ _state["cheb_coeffs"]
            if not np.all(np.isfinite(pred)):
                return np.zeros_like(y)
            return pred

    # 3. Prediction Phase
    # The evaluator will call equation with test points and params (which it fits).
    # However, our strategy is to IGNORE the evaluator's params and use our pre-computed 
    # Chebyshev coefficients? 
    # NO. The interface is equation(x, y, params). The evaluator fits params.
    # If we return X_cheb @ params, the evaluator's BFGS will fit params to minimize 
    # MSE(params, y_train). 
    # Since we pre-computed the best linear approximation (Chebyshev projection of the 
    # best non-linear model), the evaluator's BFGS will start from params=ones, 
    # but the landscape is linear and convex. It will converge to the optimal linear 
    # approximation of the target function within the Chebyshev basis.
    # This is robust because:
    # 1. We found the best *structure* (non-linear).
    # 2. We projected it to a *stable* basis (Chebyshev).
    # 3. The evaluator's BFGS just needs to find the linear coefficients for that basis, 
    #    which is a well-conditioned problem.
    
    # Build Chebyshev basis for current x
    X_cheb = _build_chebyshev_basis(x, y)
    
    if not np.all(np.isfinite(X_cheb)):
        # Fallback to simple poly if Chebyshev fails (shouldn't happen for [-1,1])
        X_cheb = np.column_stack([
            np.ones(len(x)), x, y, x**2, y**2, x*y, x**3, y**3, x**2*y, x*y**2
        ])

    # Use the params provided by the evaluator
    # The evaluator fits params to minimize MSE. 
    # Since the basis is fixed and linear, the optimal params are unique.
    # We just return the linear combination.
    
    pred = X_cheb @ params
    
    if not np.all(np.isfinite(pred)):
        return np.zeros_like(y)
        
    return pred

# EVOLVE-BLOCK-END
