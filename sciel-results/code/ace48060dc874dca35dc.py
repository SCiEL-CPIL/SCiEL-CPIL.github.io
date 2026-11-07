import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find a root x such that |f(x)| <= 1e-8 for a 1D scalar function.
    Uses a coarse scan to bracket a sign change, then Brent's method.
    Falls back to minimizing (f(x))^2 if no sign change is found.
    """
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    x0 = float(x0)
    
    # Helper to check time
    def time_left():
        return deadline - time.time()

    # Helper to check if a value is a good root
    def is_root(x, fx):
        if not np.isfinite(x):
            return False
        if not np.isfinite(fx):
            return False
        return abs(fx) <= 1e-8

    # Phase 0: Check starting point
    try:
        fx0 = f(x0)
        if is_root(x0, fx0):
            return x0
    except Exception:
        pass

    # Phase 1: Coarse scan to find a bracket [a, b] where f(a)*f(b) < 0
    # or a point with very small |f(x)|
    best_x = x0
    best_f = fx0 if 'fx0' in locals() and np.isfinite(fx0) else float('inf')
    best_abs_f = abs(best_f) if np.isfinite(best_f) else float('inf')

    # We will scan in expanding intervals around x0
    # Start with a small step, expand if no sign change found
    base_step = 1.0
    max_expansions = 10  # 1, 2, 4, ..., 1024 steps
    
    a = None
    b = None
    fa = None
    fb = None

    for exp in range(max_expansions):
        if time_left() < 1.0:
            break
        step = base_step * (2 ** exp)
        
        # Define scan range: [x0 - step, x0 + step]
        # Number of points: 21 points (10 on each side + center)
        # This is a fixed number of evaluations per expansion
        num_points = 21
        xs = np.linspace(x0 - step, x0 + step, num_points)
        
        prev_x = None
        prev_f = None
        
        for x in xs:
            if time_left() < 0.5:
                break
            try:
                fx = f(x)
            except Exception:
                continue
            
            if not np.isfinite(fx):
                prev_x = x
                prev_f = fx
                continue

            # Update best
            if abs(fx) < best_abs_f:
                best_abs_f = abs(fx)
                best_x = x
                best_f = fx
            
            # Check for sign change with previous point
            if prev_f is not None and np.isfinite(prev_f):
                if prev_f * fx < 0:
                    a = prev_x
                    b = x
                    fa = prev_f
                    fb = fx
                    break
            
            prev_x = x
            prev_f = fx
        
        if a is not None:
            break

    # If we found a bracket, use Brent's method
    if a is not None and b is not None:
        try:
            from scipy.optimize import brentq
            # brentq requires f(a) and f(b) to have opposite signs
            # We already checked this.
            # xtol=1e-12, rtol=1e-12 for high precision
            # maxiter=100 to keep call count low
            root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if np.isfinite(root):
                # Verify
                try:
                    fr = f(root)
                    if abs(fr) <= 1e-8:
                        return root
                except Exception:
                    pass
        except Exception:
            pass

    # Phase 2: If no bracket found, try minimizing (f(x))^2
    # Use the best_x found in the scan as a starting point for a local refinement
    # Or use a bounded scalar minimization around the best region
    
    try:
        from scipy.optimize import minimize_scalar
        
        def objective(x):
            try:
                fx = f(x)
                if not np.isfinite(fx):
                    return 1e100
                return fx * fx
            except Exception:
                return 1e100

        # We need a bracket for bounded method or use Nelder-Mead / Powell
        # Let's try a local refinement around best_x
        # Use a small window around best_x
        # If best_x is from a scan, we don't know the scale. 
        # Let's assume the root is within the last scanned range or use a heuristic.
        
        # If we have a best_x with reasonably small f, try to polish it with Newton
        if best_abs_f < 1.0:
            # Try Newton polish
            x = best_x
            for _ in range(20):
                if time_left() < 0.5:
                    break
                try:
                    fx = f(x)
                    if abs(fx) <= 1e-8:
                        return x
                    h = 1e-6 * max(1.0, abs(x))
                    fx_plus = f(x + h)
                    if not np.isfinite(fx_plus):
                        break
                    d = (fx_plus - fx) / h
                    if d == 0:
                        break
                    x_new = x - fx / d
                    if not np.isfinite(x_new):
                        break
                    x = x_new
                except Exception:
                    break
            return x
        
        # If best_f is large, try minimize_scalar in a wider range
        # Use the range from the last expansion
        last_step = base_step * (2 ** (max_expansions - 1))
        # But we might have broken early. Let's use a safe wide range
        # centered at x0 or best_x
        center = best_x if best_abs_f < float('inf') else x0
        width = 10.0 * max(1.0, abs(center))
        
        # Bounded minimization
        res = minimize_scalar(objective, bounds=(center - width, center + width), method='bounded', options={'xatol': 1e-12})
        if res.success:
            x = res.x
            try:
                fx = f(x)
                if abs(fx) <= 1e-8:
                    return x
            except Exception:
                pass
            return x
            
    except Exception:
        pass

    # Fallback: return best_x
    return best_x

# EVOLVE-BLOCK-END
