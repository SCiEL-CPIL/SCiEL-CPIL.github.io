import numpy as np
import time
from scipy.optimize import brentq

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 110.0
    
    def check_deadline():
        return time.time() > deadline

    x0 = float(x0)
    
    # 1. Check if x0 is already a root
    fx0 = f(x0)
    if not np.isfinite(fx0):
        # If f(x0) is bad, try a small perturbation
        x0 = x0 + 1e-6
        fx0 = f(x0)
        if not np.isfinite(fx0):
            return x0 # Can't do anything if f is broken

    if abs(fx0) <= 1e-8:
        return x0

    # 2. Primary Strategy: Secant Method
    # This is more efficient than Newton (1 eval/step vs 3) and robust for polynomials.
    # Initialize with x0 and x0 + h
    h = 1e-4 * max(1.0, abs(x0))
    x1 = x0
    f1 = fx0
    x2 = x0 + h
    f2 = f(x2)
    
    if not np.isfinite(f2):
        # Try negative perturbation
        x2 = x0 - h
        f2 = f(x2)
        if not np.isfinite(f2):
             # Fallback to simple bisection expansion if we can't even get a second point
             return _fallback_brent(f, x0, fx0, check_deadline)

    best_x = x1 if abs(f1) < abs(f2) else x2
    best_f = min(abs(f1), abs(f2))

    for i in range(50):
        if check_deadline():
            break
            
        if abs(f2) <= 1e-8:
            return x2
        if abs(f1) <= 1e-8:
            return x1

        # Check for stagnation or divergence
        if abs(x2 - x1) < 1e-14 * max(1.0, abs(x2)):
            break

        denom = f2 - f1
        if abs(denom) < 1e-15:
            break
            
        x3 = x2 - f2 * (x2 - x1) / denom
        
        if not np.isfinite(x3):
            break
            
        f3 = f(x3)
        if not np.isfinite(f3):
            break
            
        # Update best
        if abs(f3) < best_f:
            best_f = abs(f3)
            best_x = x3

        # Shift
        x1, f1 = x2, f2
        x2, f2 = x3, f3

        # Check convergence
        if abs(f3) <= 1e-8:
            return x3

    if abs(f2) <= 1e-8:
        return x2
    if abs(f1) <= 1e-8:
        return x1

    # 3. Fallback: Find a bracket and use Brent's Method
    # Use the best point found so far as the center for bracketing
    center = best_x
    fc = f(center)
    
    # If we already have a sign change in our secant history, use that
    if f1 * f2 < 0:
        a, b = x1, x2
        fa, fb = f1, f2
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
        try:
            root = brentq(f, a, b, xtol=1e-12, maxiter=50)
            if np.isfinite(root):
                return root
        except Exception:
            pass

    # Otherwise, expand from center
    radius = 1.0
    max_radius = 1e6
    found_bracket = False
    
    for _ in range(30):
        if check_deadline():
            break
            
        a_left = center - radius
        b_right = center + radius
        
        try:
            fa_left = f(a_left)
            if np.isfinite(fa_left) and fa_left * fc < 0:
                a, b = a_left, center
                fa, fb = fa_left, fc
                found_bracket = True
                break
        except Exception:
            pass
            
        try:
            fb_right = f(b_right)
            if np.isfinite(fb_right) and fb_right * fc < 0:
                a, b = center, b_right
                fa, fb = fc, fb_right
                found_bracket = True
                break
        except Exception:
            pass
        
        radius *= 2.0
        if radius > max_radius:
            break

    if found_bracket:
        try:
            root = brentq(f, a, b, xtol=1e-12, maxiter=50)
            if np.isfinite(root):
                return root
        except Exception:
            pass

    return best_x

def _fallback_brent(f, x0, fx0, check_deadline):
    # Simple bracket expansion from x0
    radius = 1.0
    max_radius = 1e6
    for _ in range(30):
        if check_deadline():
            break
        a_left = x0 - radius
        b_right = x0 + radius
        try:
            fa_left = f(a_left)
            if np.isfinite(fa_left) and fa_left * fx0 < 0:
                try:
                    root = brentq(f, a_left, x0, xtol=1e-12, maxiter=50)
                    if np.isfinite(root):
                        return root
                except Exception:
                    pass
        except Exception:
            pass
        try:
            fb_right = f(b_right)
            if np.isfinite(fb_right) and fb_right * fx0 < 0:
                try:
                    root = brentq(f, x0, b_right, xtol=1e-12, maxiter=50)
                    if np.isfinite(root):
                        return root
                except Exception:
                    pass
        except Exception:
            pass
        radius *= 2.0
        if radius > max_radius:
            break
    return x0
