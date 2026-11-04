import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import math

def _poly_moment(coeffs, x):
    """Evaluate polynomial with given coefficients at point x."""
    result = 0.0
    for c in coeffs:
        result = result * x + c
    return result

def _build_p_squared_coeffs(roots):
    """
    Build coefficients of P(x)^2 where P(x) = prod_i (x - r_i)^2
    for 12 double roots. The polynomial has degree 24.
    Returns list of coefficients [c_0, c_1, ..., c_24] such that
    P(x)^2 = sum c_k x^k.
    """
    # P(x) = prod_i (x^2 - 2*r_i*x + r_i^2)
    
    # Start with polynomial 1
    p_coeffs = [1.0]  # constant term
    
    for r in roots:
        # Multiply by (x^2 - 2r*x + r^2)
        new_coeffs = [0.0] * (len(p_coeffs) + 2)
        for i, c in enumerate(p_coeffs):
            new_coeffs[i] += c * (r * r)
            new_coeffs[i + 1] += c * (-2.0 * r)
            new_coeffs[i + 2] += c
        p_coeffs = new_coeffs
    
    # Now square P(x) to get P(x)^2
    # p_coeffs has length 13 (degree 12)
    # P(x)^2 has degree 24
    sq_coeffs = [0.0] * (2 * len(p_coeffs) - 1)
    for i in range(len(p_coeffs)):
        for j in range(len(p_coeffs)):
            sq_coeffs[i + j] += p_coeffs[i] * p_coeffs[j]
    
    return sq_coeffs

def _compute_c4(roots):
    """
    Compute the C4 upper bound for the given 12 double-root locations.
    """
    if len(roots) != 12:
        return 1e10
    
    # Check all positive
    if any(r <= 0 for r in roots):
        return 1e10
    
    try:
        sq_coeffs = _build_p_squared_coeffs(roots)
    except Exception:
        return 1e10
    
    numerator = 0.0
    denominator = 0.0
    
    # Precompute factorials up to 25
    # k! for k=0..25
    fact = [1.0] * 26
    for i in range(1, 26):
        fact[i] = fact[i-1] * i
        
    for k, c in enumerate(sq_coeffs):
        if c == 0:
            continue
        numerator += c * fact[k]
        denominator += c * fact[k+1]
    
    if denominator <= 0 or numerator <= 0:
        return 1e10
    
    c4 = math.sqrt(numerator / denominator)
    
    return c4

def _objective(y):
    """Objective function: minimize C4 bound. y = log(roots) to ensure positivity."""
    try:
        x = np.exp(y)
        if np.any(x <= 0) or np.any(np.isnan(x)) or np.any(np.isinf(x)):
            return 1e10
        c4 = _compute_c4(x)
        if not np.isfinite(c4):
            return 1e10
        return c4
    except Exception:
        return 1e10

def construct():
    """
    Constructs 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Optimizes to minimize the C4 upper bound.
    """
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 580.0  # 20s safety margin
    
    m = 12
    
    # Step 1: Get initial roots from Laguerre polynomial L_12(x)
    try:
        roots, weights = np.polynomial.laguerre.laggauss(m)
    except Exception:
        roots = 4.0 * np.arange(1, m + 1)
    
    best_roots = roots.copy()
    best_c4 = _compute_c4(roots)
    
    # Step 2: Parametric Scaling Search
    # Hypothesis: The optimal roots are a scaled version of a "standard" set.
    # We optimize a single scaling factor lambda for several base configurations.
    
    base_configs = []
    # Standard Laguerre
    base_configs.append(roots.copy())
    # Generalized Laguerre with different alphas
    try:
        from scipy.special import roots_genlaguerre
        for alpha in [0.5, 1.0, 2.0, 5.0]:
            g_roots, _ = roots_genlaguerre(m, alpha)
            base_configs.append(np.sort(g_roots))
    except Exception:
        pass
    # Uniformly spaced (log and linear)
    base_configs.append(np.exp(np.linspace(0, np.log(50), m)))
    base_configs.append(np.linspace(0.1, 50, m))
    
    for base in base_configs:
        if time.time() > deadline - 30:
            break
        
        def _scaled_obj(log_lambda):
            try:
                lam = math.exp(log_lambda[0])
                scaled_roots = base * lam
                return _compute_c4(scaled_roots)
            except Exception:
                return 1e10
        
        # Optimize log(lambda) in a reasonable range
        # Roots typically range from O(1) to O(100) for m=12
        res = minimize(_scaled_obj, [0.0], method='Nelder-Mead', 
                       options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 200})
        
        if res.success and np.isfinite(res.fun) and res.fun < best_c4:
            lam = math.exp(res.x[0])
            candidate_roots = base * lam
            if np.all(candidate_roots > 0):
                best_c4 = res.fun
                best_roots = candidate_roots
    
    # Step 3: Global Search using Differential Evolution
    # This provides a structural change to escape local minima of local optimizers.
    if time.time() < deadline - 60:
        try:
            # Bounds for log(roots)
            # Roots are positive. Let's assume range [0.01, 1000] -> log range [-4.6, 6.9]
            bounds = [(-4.6, 6.9)] * m
            
            # Use a subset of the population to keep it fast
            # DE is stochastic, so we limit iterations
            def _de_obj(y):
                return _objective(y)
            
            # Initial best is our current best_c4
            # We want to find a better starting point for L-BFGS-B
            
            # Run DE with limited iterations
            # maxiter=100, popsize=10 (10*12=120 evaluations per gen -> 12000 evals)
            # Each eval is ~1ms? 12000ms = 12s. This is safe.
            de_result = differential_evolution(
                _de_obj, bounds, 
                seed=42, 
                maxiter=50, 
                popsize=10,
                tol=1e-6,
                polish=False # We will polish manually
            )
            
            if de_result.success and np.isfinite(de_result.fun) and de_result.fun < best_c4:
                best_c4 = de_result.fun
                best_roots = np.exp(de_result.x)
            elif np.isfinite(de_result.fun) and de_result.fun < 1e9:
                # Even if not better than best, use it as a start for polish?
                # No, stick to best. But if best was from scaling, DE might have found a different basin.
                # Let's just use the DE result as a candidate for polish.
                pass
                
        except Exception:
            pass
    
    # Step 4: Local Optimization (Polish)
    # Use the best roots found so far
    if time.time() < deadline - 10:
        try:
            y0 = np.log(best_roots)
            
            # Use L-BFGS-B
            result = minimize(
                _objective, 
                y0, 
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            if np.isfinite(result.fun) and result.fun < best_c4:
                optimal_roots = np.exp(result.x)
                if np.all(optimal_roots > 0):
                    best_roots = optimal_roots
                    best_c4 = result.fun
                    
        except Exception:
            pass
            
    # Step 5: One more polish with Nelder-Mead if time permits and L-BFGS didn't improve much
    if time.time() < deadline - 5:
        try:
            y0 = np.log(best_roots)
            result2 = minimize(
                _objective,
                y0,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if np.isfinite(result2.fun) and result2.fun < best_c4:
                optimal_roots = np.exp(result2.x)
                if np.all(optimal_roots > 0):
                    best_roots = optimal_roots
                    best_c4 = result2.fun
        except Exception:
            pass
    
    return best_roots.tolist()
