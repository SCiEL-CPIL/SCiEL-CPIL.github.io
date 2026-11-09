import numpy as np
from scipy.special import roots_genlaguerre
from scipy.optimize import minimize
import time

_START_TIME = time.time()
_TIME_LIMIT = 550.0  # Safety margin before 600s limit

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return max(0.0, _TIME_LIMIT - _elapsed())

# Pre-compute Gauss-Laguerre nodes and weights for high accuracy
# Use higher N for better numerical stability in log-space
_GL_N = 30
try:
    _gl_nodes, _gl_weights = np.polynomial.laguerre.laggauss(_GL_N)
except Exception:
    _gl_nodes = np.linspace(0, 100, _GL_N)
    _gl_weights = np.ones(_GL_N) / _GL_N

def _compute_l2_norm_sq(roots):
    """
    Compute the square of the L2 norm of P(x) = prod (x - r_i)^2 with weight e^-x.
    Uses Gauss-Laguerre quadrature for speed.
    """
    roots = np.asarray(roots, dtype=float)
    if len(roots) != 12:
        return 1e100
    if np.any(roots <= 0):
        return 1e100
    
    try:
        diff = _gl_nodes[:, np.newaxis] - roots[np.newaxis, :]
        abs_diff = np.abs(diff)
        # Avoid log(0)
        abs_diff[abs_diff == 0] = 1e-300
        log_p_sq = 4.0 * np.sum(np.log(abs_diff), axis=1)
        
        log_w = np.log(_gl_weights)
        log_terms_w = log_p_sq + log_w
        
        max_log = np.max(log_terms_w)
        if np.isnan(max_log):
             return 1e100
             
        if max_log > 700:
             return 1e100
             
        sum_exp = np.sum(np.exp(log_terms_w - max_log))
        result = np.exp(max_log) * sum_exp
        return result
        
    except Exception:
        return 1e100

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Score is the upper bound on C4 (lower is better).
    """
    m = 12
    
    # Generate multiple initial guesses to escape local minima
    initial_guesses = []
    
    # 1. Roots of generalized Laguerre polynomial L_12^(0)(x)
    try:
        roots_lag = roots_genlaguerre(m, 0)
        roots_lag = np.sort(roots_lag.real)
        if len(roots_lag) == 12 and np.all(roots_lag > 0):
            initial_guesses.append(roots_lag.copy())
    except Exception:
        pass
    
    # 2. Uniformly spaced roots
    initial_guesses.append(np.array([4.0 * (i + 1) for i in range(12)]))
    
    # 3. Geometrically spaced roots
    initial_guesses.append(np.geomspace(1.0, 50.0, 12))
    
    # 4. Roots perturbed from Laguerre roots
    if initial_guesses:
        try:
            roots_perturbed = initial_guesses[0] * (1.0 + 0.1 * np.random.randn(12))
            roots_perturbed = np.sort(np.abs(roots_perturbed) + 0.1)
            if len(roots_perturbed) == 12 and np.all(roots_perturbed > 0):
                initial_guesses.append(roots_perturbed)
        except Exception:
            pass
    
    best_roots = None
    best_score = float('inf')
    
    # Optimize from each initial guess
    for i, roots0 in enumerate(initial_guesses):
        if _remaining() < 15.0:
            break
        
        try:
            score0 = _compute_l2_norm_sq(roots0)
            if score0 < best_score:
                best_score = score0
                best_roots = roots0.copy()
            
            # Optimize in log-space to ensure positivity
            y0 = np.log(roots0)
            bounds_y = [(-15.0, 15.0) for _ in range(12)]
            
            def objective_log(y, best_score_ref=[best_score]):
                r = np.exp(y)
                return _compute_l2_norm_sq(r)
            
            # First pass: L-BFGS-B
            result = minimize(
                objective_log,
                y0,
                method='L-BFGS-B',
                bounds=bounds_y,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_roots = np.exp(result.x)
            
            # Second pass: polish with tighter tolerance
            if _remaining() > 20.0 and result.fun < best_score + 1e-6:
                result2 = minimize(
                    objective_log,
                    result.x,
                    method='L-BFGS-B',
                    bounds=bounds_y,
                    options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10, 'disp': False}
                )
                if result2.fun < best_score:
                    best_score = result2.fun
                    best_roots = np.exp(result2.x)
                    
        except Exception:
            continue
    
    # Final high-precision polish if we have a good candidate
    if best_roots is not None and _remaining() > 30.0:
        try:
            y_final = np.log(best_roots)
            bounds_y = [(-15.0, 15.0) for _ in range(12)]
            
            def objective_final(y):
                r = np.exp(y)
                return _compute_l2_norm_sq(r)
            
            result_final = minimize(
                objective_final,
                y_final,
                method='L-BFGS-B',
                bounds=bounds_y,
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-12, 'disp': False}
            )
            if result_final.fun < best_score:
                best_score = result_final.fun
                best_roots = np.exp(result_final.x)
        except Exception:
            pass
    
    # Fallback if no valid roots found
    if best_roots is None:
        best_roots = np.array([4.0 * (i + 1) for i in range(12)])
    
    # Final validation
    best_roots = np.sort(best_roots)
    if len(best_roots) != 12 or np.any(best_roots <= 0) or np.any(np.diff(best_roots) < 1e-8):
        best_roots = np.array([4.0 * (i + 1) for i in range(12)])
    
    return [float(r) for r in best_roots]
