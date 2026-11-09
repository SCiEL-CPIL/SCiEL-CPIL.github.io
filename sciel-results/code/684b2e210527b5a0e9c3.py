import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 580  # 580s safety margin before 630s kill
    
    m = 12
    n = 1
    
    def compute_c4(roots):
        """Compute C4 bound efficiently using coarse-fine grid."""
        roots = np.asarray(roots, dtype=np.float64)
        if len(roots) != m:
            return np.inf
        if np.any(roots <= 0.001) or np.any(np.diff(roots) < 0.0001):
            return np.inf
        
        x_max = max(roots[-1] + 5.0, 20.0)
        
        # Coarse grid
        x_coarse = np.linspace(0.01, x_max, 1500)
        log_f_coarse = np.zeros_like(x_coarse)
        for r in roots:
            log_f_coarse += 2.0 * np.log(np.abs(x_coarse - r) + 1e-30)
        log_f_coarse -= 0.5 * x_coarse
        log_f_coarse -= 0.5 * n * np.log(x_coarse + 1e-30)
        
        idx = np.argmax(log_f_coarse)
        
        # Refine around the max
        if idx > 0 and idx < len(x_coarse) - 1:
            x_left = max(x_coarse[idx-1], 0.005)
            x_right = min(x_coarse[idx+1], x_max)
        else:
            x_left = max(0.005, x_coarse[idx] * 0.5)
            x_right = min(x_coarse[idx] * 2.0, x_max)
        
        x_fine = np.linspace(x_left, x_right, 800)
        log_f_fine = np.zeros_like(x_fine)
        for r in roots:
            log_f_fine += 2.0 * np.log(np.abs(x_fine - r) + 1e-30)
        log_f_fine -= 0.5 * x_fine
        log_f_fine -= 0.5 * n * np.log(x_fine + 1e-30)
        
        max_log_f = np.max(log_f_fine)
        
        # Check intervals between roots for local maxima
        for i in range(len(roots) - 1):
            a = max(roots[i] + 0.01, 0.01)
            b = max(roots[i+1] - 0.01, a + 0.01)
            if b > a:
                x_check = np.linspace(a, b, 150)
                log_f_check = np.zeros_like(x_check)
                for r in roots:
                    log_f_check += 2.0 * np.log(np.abs(x_check - r) + 1e-30)
                log_f_check -= 0.5 * x_check
                log_f_check -= 0.5 * n * np.log(x_check + 1e-30)
                max_log_f = max(max_log_f, np.max(log_f_check))
        
        return np.exp(max_log_f)
    
    def objective(roots_flat):
        roots = np.asarray(roots_flat, dtype=np.float64)
        if np.any(roots <= 0.01) or np.any(np.diff(roots) < 0.001):
            return 1e10
        return compute_c4(roots)
    
    # Structural change: use Laguerre polynomial roots as starting points
    try:
        from scipy.special import roots_laguerre
        from scipy.optimize import minimize
        import warnings
        warnings.filterwarnings('ignore')
        
        # Get roots of Laguerre polynomial L_m
        lag_roots, _ = roots_laguerre(m)
        
        best_roots = lag_roots.copy()
        best_score = objective(best_roots)
        
        # Try different transformations and random starts
        candidates = [
            lag_roots,
            lag_roots * 2.0,
            lag_roots * 0.5,
            lag_roots + 1.0,
            lag_roots + 2.0,
            np.linspace(0.5, 12.0, m),
            np.linspace(1.0, 15.0, m),
        ]
        
        np.random.seed(42)
        for _ in range(5):
            candidates.append(np.sort(np.random.uniform(0.5, 20.0, m)))
        
        bounds = [(0.01, 50.0)] * m
        
        for i, x0 in enumerate(candidates):
            if time.time() > deadline - 30:
                break
            try:
                res = minimize(
                    objective, x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.fun < best_score:
                    best_score = res.fun
                    best_roots = res.x
            except Exception:
                pass
        
    except Exception:
        pass
    
    # Ensure roots are positive and sorted
    best_roots = np.sort(np.asarray(best_roots, dtype=np.float64))
    best_roots = np.maximum(best_roots, 0.01)
    
    # Final validation and correction
    for i in range(len(best_roots) - 1):
        if best_roots[i+1] <= best_roots[i] + 0.001:
            best_roots[i+1] = best_roots[i] + 0.001
    
    # Convert to list
    return [float(r) for r in best_roots]
