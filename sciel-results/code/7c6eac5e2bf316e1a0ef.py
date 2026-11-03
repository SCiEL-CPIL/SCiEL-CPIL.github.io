import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
from math import factorial

def _compute_c4_exact(roots):
    """
    Compute the C4 bound exactly using polynomial moments.
    C4 = ( \int P(x)^2 e^{-x} dx ) / ( \int P(x) e^{-x} dx )^2
    where P(x) = prod (x - r_i)^2
    """
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.inf

    # Construct Q(x) = prod (x - r_i)
    # Coefficients in descending order
    q_coeffs = np.poly(roots)
    
    # P(x) = Q(x)^2
    # Convolve q_coeffs with itself
    p_coeffs = np.convolve(q_coeffs, q_coeffs)
    
    # p_coeffs[k] is coefficient of x^(24-k)
    # We need \int x^j e^{-x} dx = j!
    # Let index j correspond to power of x.
    # p_coeffs[0] is coeff of x^24, p_coeffs[1] of x^23, ..., p_coeffs[24] of x^0.
    
    # I_0 = \int P(x) e^{-x} dx = \sum_{j=0}^{24} p_coeffs[24-j] * j!
    # Let's map correctly.
    # p_coeffs[i] corresponds to power (24 - i).
    # So for power j, index i = 24 - j.
    # I_0 = \sum_{j=0}^{24} p_coeffs[24-j] * factorial(j)
    
    # Precompute factorials
    max_deg = len(p_coeffs) - 1 # 24
    facts = np.array([factorial(j) for j in range(max_deg + 1)], dtype=np.float64)
    
    # Reverse p_coeffs to align with increasing powers
    # p_coeffs_rev[j] is coeff of x^j
    p_coeffs_rev = p_coeffs[::-1]
    
    I_0 = np.dot(p_coeffs_rev, facts)
    
    # I_1 = \int P(x)^2 e^{-x} dx
    # P(x)^2 = (Q(x)^2)^2 = Q(x)^4
    # Alternatively, convolve p_coeffs with p_coeffs
    p2_coeffs = np.convolve(p_coeffs, p_coeffs)
    # p2_coeffs has length 49 (powers 48 down to 0)
    p2_coeffs_rev = p2_coeffs[::-1]
    
    # Need factorials up to 48
    facts_48 = np.array([factorial(j) for j in range(49)], dtype=np.float64)
    
    I_1 = np.dot(p2_coeffs_rev, facts_48)
    
    if I_0 <= 0 or np.isinf(I_0) or np.isinf(I_1):
        return np.inf
        
    return I_1 / (I_0 * I_0)

def _objective_log_y(y):
    """Objective function for optimization in log-space."""
    roots = np.exp(y)
    return _compute_c4_exact(roots)

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    deadline = time.time() + 580  # Safety margin before 630s kill
    
    # Initial guess: Laguerre roots
    from scipy.special import roots_laguerre
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    
    # Convert to log-space for optimization
    y0 = np.log(x0)
    
    best_y = y0.copy()
    best_score = _objective_log_y(best_y)
    
    # Optimization Strategy:
    # 1. Differential Evolution in log-space (unconstrained)
    # 2. Local polish with L-BFGS-B in log-space
    
    # DE bounds for log(y):
    # Roots likely in [0.1, 100] -> log in [-2.3, 4.6]
    # Let's use [-3, 5] to be safe.
    bounds = [(-3.0, 5.0)] * 12
    
    # Run Differential Evolution
    # Using workers=-1 might help if multi-core, but let's stick to 1 for determinism/simplicity unless speed is critical.
    # 600s is plenty for 12 dims.
    
    try:
        de_result = differential_evolution(
            _objective_log_y,
            bounds=bounds,
            maxiter=100,
            popsize=15,
            tol=1e-8,
            seed=42,
            polish=False, # We will polish manually for better control
            updating='immediate',
            workers=1
        )
        
        if de_result.fun < best_score:
            best_score = de_result.fun
            best_y = de_result.x.copy()
            
    except Exception as e:
        print(f"DE failed: {e}")
        
    # Polish the best candidate found so far
    if time.time() < deadline - 30:
        try:
            polish_result = minimize(
                _objective_log_y,
                best_y,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if polish_result.fun < best_score:
                best_score = polish_result.fun
                best_y = polish_result.x.copy()
        except Exception as e:
            print(f"Polish failed: {e}")
            
    # Try a second DE run with different seed and tighter bounds if time permits
    if time.time() < deadline - 60:
        try:
            # Tighter bounds based on previous result
            center = best_y
            radius = 1.0
            bounds_tight = [(center[i] - radius, center[i] + radius) for i in range(12)]
            
            de_result2 = differential_evolution(
                _objective_log_y,
                bounds=bounds_tight,
                maxiter=50,
                popsize=15,
                tol=1e-10,
                seed=123,
                polish=False,
                updating='immediate',
                workers=1
            )
            
            if de_result2.fun < best_score:
                best_score = de_result2.fun
                best_y = de_result2.x.copy()
                
            # Final polish
            if time.time() < deadline - 10:
                polish_result2 = minimize(
                    _objective_log_y,
                    best_y,
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14}
                )
                if polish_result2.fun < best_score:
                    best_score = polish_result2.fun
                    best_y = polish_result2.x.copy()
                    
        except Exception as e:
            print(f"Second DE/Polish failed: {e}")
            
    # Convert back to linear space
    best_roots = np.exp(best_y)
    
    return [float(xi) for xi in best_roots]
