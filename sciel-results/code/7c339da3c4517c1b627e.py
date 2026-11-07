"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import brentq, minimize_scalar

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    TIME_LIMIT = 145.0  # safety margin before 150s kill
    x0 = float(x0)
    
    # Check initial point
    try:
        f0 = f(x0)
    except:
        f0 = float('inf')
    
    if not np.isfinite(f0):
        f0 = float('inf')
    
    if abs(f0) <= 1e-8:
        return x0
    
    best_x = x0
    best_f = abs(f0)
    
    def time_ok():
        return (time.time() - start_time) < TIME_LIMIT
    
    def update_best(x, fx):
        nonlocal best_x, best_f
        if np.isfinite(fx) and abs(fx) < best_f:
            best_x = x
            best_f = abs(fx)
            if best_f <= 1e-8:
                return True
        return False
    
    # Strategy 1: Try Newton's method with multiple starting points
    # This is fast if it converges
    def newton_from(x_start, max_iter=50):
        if not time_ok():
            return None
        x = float(x_start)
        for i in range(max_iter):
            if not time_ok():
                return None
            try:
                fx = f(x)
            except:
                return None
            if not np.isfinite(fx):
                return None
            if abs(fx) <= 1e-8:
                return x
            h = 1e-7 * max(1.0, abs(x))
            try:
                fx_plus = f(x + h)
            except:
                return None
            if not np.isfinite(fx_plus):
                return None
            d = (fx_plus - fx) / h
            if d == 0 or not np.isfinite(d):
                return None
            x_new = x - fx / d
            if not np.isfinite(x_new):
                return None
            # Damping if step is too large
            if abs(x_new - x) > 10 * max(1.0, abs(x)):
                x_new = x + np.sign(x_new - x) * 10 * max(1.0, abs(x))
            x = x_new
        try:
            fx = f(x)
            if np.isfinite(fx) and abs(fx) <= 1e-8:
                return x
        except:
            pass
        return None
    
    # Try Newton from x0 and a few nearby points
    newton_starts = [x0, x0 + 0.1, x0 - 0.1, x0 + 1.0, x0 - 1.0, x0 + 0.5, x0 - 0.5]
    for xs in newton_starts:
        if not time_ok():
            break
        result = newton_from(xs)
        if result is not None:
            try:
                fx = f(result)
                if np.isfinite(fx) and abs(fx) <= 1e-8:
                    return result
            except:
                pass
    
    # Strategy 2: Sign-change scan at multiple scales, then Brent's method
    # This handles oscillatory functions well
    scales = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    
    for scale in scales:
        if not time_ok():
            break
        
        # Scan in both directions
        for direction in [1, -1]:
            if not time_ok():
                break
            
            # Number of steps to try
            n_steps = 200
            x_prev = x0
            try:
                f_prev = f(x_prev)
            except:
                f_prev = float('inf')
            
            if not np.isfinite(f_prev):
                continue
                
            for i in range(1, n_steps + 1):
                if not time_ok():
                    break
                
                x_curr = x0 + direction * i * scale
                try:
                    f_curr = f(x_curr)
                except:
                    break
                
                if not np.isfinite(f_curr):
                    break
                    
                if np.isfinite(f_prev) and f_prev * f_curr <= 0:
                    # Sign change found, use Brent's method
                    try:
                        root = brentq(f, x_prev, x_curr, xtol=1e-14, rtol=1e-14, maxiter=100)
                        if np.isfinite(root):
                            fx = f(root)
                            if np.isfinite(fx) and abs(fx) <= 1e-8:
                                return root
                    except:
                        pass
                
                x_prev = x_curr
                f_prev = f_curr
    
    # Strategy 3: Nelder-Mead on |f(x)|
    if time_ok():
        def objective(x):
            try:
                val = f(float(x[0]))
                if not np.isfinite(val):
                    return 1e30
                return abs(val)
            except:
                return 1e30
        
        # Try multiple starting points for Nelder-Mead
        nm_starts = [x0, x0 + 1.0, x0 - 1.0, x0 + 0.1, x0 - 0.1]
        
        for xs in nm_starts:
            if not time_ok():
                break
            
            try:
                res = minimize_scalar(objective, bounds=(xs - 10.0, xs + 10.0), method='bounded', options={'xatol': 1e-12})
                if res.success and np.isfinite(res.fun) and res.fun <= 1e-8:
                    x_root = float(res.x[0])
                    fx = f(x_root)
                    if np.isfinite(fx) and abs(fx) <= 1e-8:
                        return x_root
            except:
                pass
            
            # Also try Nelder-Mead without bounds
            try:
                res = minimize_scalar(objective, method='nelder-mead', options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 1000, 'maxfev': 1000})
                if np.isfinite(res.fun) and res.fun <= 1e-8:
                    x_root = float(res.x[0])
                    fx = f(x_root)
                    if np.isfinite(fx) and abs(fx) <= 1e-8:
                        return x_root
            except:
                pass
    
    # Return best found
    return best_x
# EVOLVE-BLOCK-END
