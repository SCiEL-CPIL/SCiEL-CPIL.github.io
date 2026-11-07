"""Root finder for 3 fixed oscillatory equations."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    TIME_LIMIT = 135.0  # safety margin before 150s kill
    
    x0 = float(x0)
    
    def time_ok():
        return (time.time() - start_time) < TIME_LIMIT
    
    # Check initial point
    try:
        f0 = f(x0)
    except Exception:
        f0 = float('inf')
    
    if not np.isfinite(f0):
        f0 = float('inf')
    
    if abs(f0) <= 1e-8:
        return x0
    
    best_x = x0
    best_f = abs(f0)
    
    # Helper: safe function evaluation
    def safe_f(x):
        try:
            val = f(x)
            if not np.isfinite(val):
                return float('inf')
            return float(val)
        except Exception:
            return float('inf')
    
    # Helper: try to refine a sign-change bracket [a, b] using bisection
    def bisect(a, b, fa, fb, max_iter=100):
        nonlocal best_x, best_f
        if fa * fb > 0:
            return None
        for _ in range(max_iter):
            if not time_ok():
                break
            m = (a + b) / 2.0
            fm = safe_f(m)
            if abs(fm) < best_f:
                best_x = m
                best_f = abs(fm)
            if abs(fm) <= 1e-10:
                return m
            if fa * fm <= 0:
                b = m
                fb = fm
            else:
                a = m
                fa = fm
            if abs(b - a) < 1e-14 * max(1.0, abs(a)):
                break
        return (a + b) / 2.0
    
    # Helper: Newton step with safeguarding
    def newton_step(x, fx):
        h = 1e-7 * max(1.0, abs(x))
        fxh = safe_f(x + h)
        if not np.isfinite(fxh):
            return None
        d = (fxh - fx) / h
        if d == 0 or not np.isfinite(d):
            return None
        x_new = x - fx / d
        if not np.isfinite(x_new):
            return None
        # Limit step size to avoid wild jumps
        max_step = 10.0 * max(1.0, abs(x))
        if abs(x_new - x) > max_step:
            x_new = x + max_step * (1 if x_new > x else -1)
        return x_new
    
    # Strategy 1: Multi-scale sign-change scan
    # For oscillatory functions, we need to find brackets where sign changes
    scales = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    
    for scale in scales:
        if not time_ok():
            break
        
        # Scan in both directions from x0
        for direction in [1, -1]:
            if not time_ok():
                break
            
            # Number of points to scan
            n_points = 20
            
            prev_x = x0
            prev_f = f0
            
            for i in range(1, n_points + 1):
                if not time_ok():
                    break
                
                x_new = x0 + direction * i * scale
                f_new = safe_f(x_new)
                
                if np.isfinite(f_new):
                    if abs(f_new) < best_f:
                        best_x = x_new
                        best_f = abs(f_new)
                    
                    # Check for sign change
                    if np.isfinite(prev_f) and prev_f * f_new < 0:
                        root = bisect(prev_x, x_new, prev_f, f_new)
                        if root is not None and abs(safe_f(root)) <= 1e-8:
                            return root
                    
                    # Also check if f_new is very close to zero
                    if abs(f_new) <= 1e-8:
                        return x_new
                
                prev_x = x_new
                prev_f = f_new
    
    # Strategy 2: Newton's method from best point
    if time_ok():
        x = best_x
        fx = safe_f(x)
        for _ in range(50):
            if not time_ok():
                break
            if abs(fx) <= 1e-10:
                return x
            x_new = newton_step(x, fx)
            if x_new is None:
                break
            fx_new = safe_f(x_new)
            if abs(fx_new) < abs(fx):
                x = x_new
                fx = fx_new
                if abs(fx) < best_f:
                    best_x = x
                    best_f = abs(fx)
            else:
                # Halve the step
                x_mid = (x + x_new) / 2.0
                fx_mid = safe_f(x_mid)
                if abs(fx_mid) < abs(fx):
                    x = x_mid
                    fx = fx_mid
                else:
                    break
    
    # Strategy 3: Secant method from best point
    if time_ok() and best_f > 1e-8:
        # Use two nearby points
        x1 = best_x
        f1 = safe_f(x1)
        dx = 1e-4 * max(1.0, abs(x1))
        x2 = x1 + dx
        f2 = safe_f(x2)
        
        if np.isfinite(f1) and np.isfinite(f2) and f1 != f2:
            for _ in range(50):
                if not time_ok():
                    break
                if abs(f1) <= 1e-10:
                    return x1
                if abs(f2) <= 1e-10:
                    return x2
                
                # Secant step
                x_new = x1 - f1 * (x1 - x2) / (f1 - f2)
                if not np.isfinite(x_new):
                    break
                
                f_new = safe_f(x_new)
                if not np.isfinite(f_new):
                    break
                
                if abs(f_new) < best_f:
                    best_x = x_new
                    best_f = abs(f_new)
                
                if abs(f_new) <= 1e-8:
                    return x_new
                
                # Update points
                x1, f1 = x2, f2
                x2, f2 = x_new, f_new
    
    # Strategy 4: Broader scan with finer steps if still not found
    if time_ok() and best_f > 1e-8:
        # Try a denser scan around x0
        for scale in [0.001, 0.01, 0.1]:
            if not time_ok():
                break
            for direction in [1, -1]:
                if not time_ok():
                    break
                n_points = 50
                prev_x = x0
                prev_f = f0
                for i in range(1, n_points + 1):
                    if not time_ok():
                        break
                    x_new = x0 + direction * i * scale
                    f_new = safe_f(x_new)
                    if np.isfinite(f_new):
                        if abs(f_new) < best_f:
                            best_x = x_new
                            best_f = abs(f_new)
                        if np.isfinite(prev_f) and prev_f * f_new < 0:
                            root = bisect(prev_x, x_new, prev_f, f_new)
                            if root is not None and abs(safe_f(root)) <= 1e-8:
                                return root
                        if abs(f_new) <= 1e-8:
                            return x_new
                    prev_x = x_new
                    prev_f = f_new
    
    return best_x
# EVOLVE-BLOCK-END
