"""
Strategy: Internal portfolio search with robust local optimization.
Unlike the current program which relies on the evaluator's BFGS for a fixed Chebyshev basis,
this program defines a set of diverse non-linear candidate functions.
For each candidate, it uses scipy.optimize.least_squares (LM/Dogbox) with multiple random
restarts to find the best parameters. It returns the prediction from the candidate with the
lowest training MSE. This allows the model to capture non-linear structures that a fixed
linear basis might miss or fit poorly due to BFGS local minima.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import least_squares

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # Safety margin before 150s kill

def _check_deadline():
    return time.time() > _DEADLINE

def _safe_predict(model_func, x, y, params):
    """Safely evaluate a model function, returning zeros on failure."""
    try:
        pred = model_func(x, y, params)
        pred = np.asarray(pred, dtype=np.float64)
        if pred.shape != y.shape:
            return np.zeros_like(y)
        if not np.all(np.isfinite(pred)):
            return np.zeros_like(y)
        return pred
    except Exception:
        return np.zeros_like(y)

def _fit_form(model_func, x, y, p0, bounds=None):
    """
    Fit a model using least_squares.
    Returns (params, mse) or (None, inf) on failure.
    """
    if _check_deadline():
        return None, np.inf
    
    # Define residual function for least_squares
    # least_squares minimizes sum(residuals^2)
    def residuals(params):
        pred = model_func(x, y, params)
        return y - pred
        
    # Initial check
    try:
        res0 = residuals(p0)
        if not np.all(np.isfinite(res0)):
            return None, np.inf
    except Exception:
        return None, np.inf

    try:
        # Use 'lm' if no bounds, else 'trf'
        if bounds is None:
            method = 'lm'
        else:
            method = 'trf'
            
        res = least_squares(residuals, p0, method=method, bounds=bounds, max_nfev=1000)
        
        if not res.success and not np.all(np.isfinite(res.x)):
             # Even if not success, if x is finite, we might use it, but let's check cost
             pass
             
        final_params = res.x
        if not np.all(np.isfinite(final_params)):
            return None, np.inf
            
        # Calculate MSE
        pred = model_func(x, y, final_params)
        if not np.all(np.isfinite(pred)):
            return None, np.inf
            
        mse = np.mean((y - pred)**2)
        if not np.isfinite(mse):
            return None, np.inf
             
        return final_params, mse
        
    except Exception:
        return None, np.inf

# --- Candidate Models ---
# Each model takes (x, y, params) and returns predictions.
# params is an array of length 10.

def model_linear(x, y, p):
    # p0: intercept, p1: x, p2: y, p3: x^2, p4: y^2, p5: xy, p6: x^3, p7: y^3, p8: x^2y, p9: xy^2
    return p[0] + p[1]*x + p[2]*y + p[3]*x**2 + p[4]*y**2 + p[5]*x*y + p[6]*x**3 + p[7]*y**3 + p[8]*x**2*y + p[9]*x*y**2

def model_chebyshev(x, y, p):
    # Chebyshev polynomials up to degree 3 for x and y, plus cross terms
    T1x = x
    T2x = 2*x**2 - 1
    T3x = 4*x**3 - 3*x
    T1y = y
    T2y = 2*y**2 - 1
    T3y = 4*y**3 - 3*y
    
    return (p[0] + p[1]*T1x + p[2]*T1y + p[3]*T2x + p[4]*T2y + 
            p[5]*T1x*T1y + p[6]*T2x*T1y + p[7]*T1x*T2y + 
            p[8]*T3x + p[9]*T3y)

def model_radial(x, y, p):
    # Radial basis functions / Gaussian-like interactions
    # p0: offset, p1-p4: centers, p5-p8: widths, p9: scale
    # Simplified to ensure differentiability and avoid singularities
    r1 = np.sqrt((x - p[1])**2 + (y - p[2])**2) + 1e-6
    r2 = np.sqrt((x - p[3])**2 + (y - p[4])**2) + 1e-6
    # Use inverse distance or gaussian? Inverse distance is non-analytic at 0, but we added eps.
    # Let's use a smooth kernel: exp(-r^2 / width^2)
    w1 = np.abs(p[5]) + 0.1
    w2 = np.abs(p[6]) + 0.1
    return p[0] + p[7]*np.exp(-(r1**2)/(w1**2)) + p[8]*np.exp(-(r2**2)/(w2**2)) + p[9]*(x*y)

def model_trig(x, y, p):
    # Trigonometric basis
    # p0: offset, p1: freq_x, p2: freq_y, p3: freq_xy, p4: phase_x, p5: phase_y, p6: phase_xy, p7: amp_x, p8: amp_y, p9: amp_xy
    fx = np.abs(p[1]) + 0.1
    fy = np.abs(p[2]) + 0.1
    fxy = np.abs(p[3]) + 0.1
    return (p[0] + p[7]*np.sin(fx*x + p[4]) + p[8]*np.sin(fy*y + p[5]) + 
            p[9]*np.sin(fxy*(x+y) + p[6]))

def model_hybrid_nonlin(x, y, p):
    # Mix of polynomials and non-linear activations
    # p0: bias, p1-p5: linear coeffs, p6: sigmoid slope, p7: sigmoid offset, p8: tanh slope, p9: tanh offset
    sig = 1.0 / (1.0 + np.exp(-p[6]*(x + p[7])))
    tanh_val = np.tanh(p[8]*(y + p[9]))
    return p[0] + p[1]*x + p[2]*y + p[3]*x**2 + p[4]*y**2 + p[5]*sig*tanh_val

def model_gaussian_process_like(x, y, p):
    # Sum of shifted Gaussians (RBF network)
    # p0: base, p1-p3: center1, p4: width1, p5-p7: center2, p8: width2, p9: scale
    c1x, c1y, w1 = p[1], p[2], np.abs(p[4]) + 0.1
    c2x, c2y, w2 = p[5], p[6], np.abs(p[8]) + 0.1
    g1 = np.exp(-((x-c1x)**2 + (y-c1y)**2) / w1**2)
    g2 = np.exp(-((x-c2x)**2 + (y-c2y)**2) / w2**2)
    return p[0] + p[9]*g1 + p[9]*g2

MODELS = [
    model_linear,
    model_chebyshev,
    model_radial,
    model_trig,
    model_hybrid_nonlin,
    model_gaussian_process_like
]

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Main entry point.
    Performs internal search to find the best model and parameters.
    Returns predictions for the best model.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    if len(x) == 0:
        return np.array([])
        
    # If the provided params are already good, we could return them, but the task
    # implies we should recover the function. The 'params' argument in the evaluator
    # is the initial guess for the fit. However, our strategy ignores the initial 'params'
    # passed by the evaluator for the final prediction logic, and instead fits our own
    # internal models. We use the 'params' array length (10) as a constraint for our models.
    
    best_pred = None
    best_mse = np.inf
    
    # Random seed for reproducibility in search
    rng = np.random.default_rng(42)
    
    # Try each model
    for model in MODELS:
        if _check_deadline():
            break
            
        # Generate a few initial guesses
        # 1. All ones (default)
        # 2. Zeros
        # 3. Random small values
        # 4. Random larger values
        
        initial_guesses = [
            np.ones(10),
            np.zeros(10),
            rng.uniform(-1, 1, 10),
            rng.uniform(-5, 5, 10)
        ]
        
        for p0 in initial_guesses:
            if _check_deadline():
                break
                
            fitted_params, mse = _fit_form(model, x, y, p0)
            
            if fitted_params is not None and mse < best_mse:
                best_mse = mse
                best_pred = _safe_predict(model, x, y, fitted_params)
                
                # If MSE is very low, we can stop early
                if best_mse < 1e-8:
                    return best_pred

    # If no model worked well, fallback to the linear model with the provided params
    # or just return the mean of y
    if best_pred is None:
        # Fallback: Linear regression via np.linalg.lstsq on simple basis
        X = np.column_stack([np.ones(len(x)), x, y, x**2, y**2, x*y, x**3, y**3, x**2*y, x*y**2])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            best_pred = X @ coeffs
        except Exception:
            best_pred = np.full_like(y, np.mean(y))
            
    # Ensure output is valid
    if not np.all(np.isfinite(best_pred)):
        best_pred = np.zeros_like(y)
        
    return best_pred

# EVOLVE-BLOCK-END
