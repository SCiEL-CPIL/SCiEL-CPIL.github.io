"""Root finder for 3 fixed oscillatory equations."""
import time
import math

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    """
    Find a root of f starting near x0.
    Strategy:
      1. Try Newton's method for a few iterations (fast if it works).
      2. If Newton fails or stalls, bracket a sign change and use Brent's method.
    Track elapsed time to stay within deadline.
    """
    start_time = time.time()
    DEADLINE = 120.0  # seconds total budget; use safety margin
    
    def time_ok():
        return (time.time() - start_time) < (DEADLINE - 2.0)
    
    x = float(x0)
    best_x = x
    best_abs = float('inf')
    
    # Phase 1: Newton's method with finite-difference derivative
    # Use a limited number of iterations to avoid wasting calls
    for i in range(20):
        if not time_ok():
            break
        try:
            fx = f(x)
        except Exception:
            break
        if not np.isfinite(fx):
            break
        afx = abs(fx)
        if afx < best_abs:
            best_abs = afx
            best_x = x
        if afx <= 1e-12:
            return x
        if afx <= 1e-8:
            # Good enough, but try to polish a bit more if we have time
            # Actually, return immediately to save calls
            return x
        
        h = 1e-7 * max(1.0, abs(x))
        try:
            fx_plus = f(x + h)
        except Exception:
            break
        if not np.isfinite(fx_plus):
            break
        
        d = (fx_plus - fx) / h
        if d == 0 or not np.isfinite(d):
            break
        
        x_new = x - fx / d
        
        # Check if step is reasonable
        if not np.isfinite(x_new):
            break
        # If step is too large relative to x, damp it
        if abs(x_new - x) > 10.0 * max(1.0, abs(x)):
            x_new = x + math.copysign(10.0 * max(1.0, abs(x)), x_new - x)
        
        x = x_new
    
    if not time_ok():
        return best_x
    
    # Phase 2: Bracket a sign change
    # Try expanding from x0 to find a bracket
    a = x0
    fa = None
    try:
        fa = f(a)
    except Exception:
        pass
    if fa is not None and np.isfinite(fa):
        if abs(fa) <= 1e-8:
            return a
    
    # Try several scales to find a sign change
    found_bracket = False
    for scale in [1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5]:
        if not time_ok():
            break
        for direction in [1, -1]:
            if not time_ok():
                break
            b = x0 + direction * scale * max(1.0, abs(x0))
            try:
                fb = f(b)
            except Exception:
                continue
            if not np.isfinite(fb):
                continue
            if abs(fb) <= 1e-8:
                return b
            if fa is not None and np.isfinite(fa) and fa * fb < 0:
                # Sign change found
                found_bracket = True
                try:
                    root = brentq(f, a, b, xtol=1e-14, rtol=1e-14, maxiter=100)
                    if np.isfinite(root):
                        return root
                except Exception:
                    pass
                break
        if found_bracket:
            break
    
    # If no bracket found with simple scaling, try a more systematic search
    if not time_ok():
        return best_x
    
    # Try searching in a grid around x0
    for log_scale in range(-4, 7):
        if not time_ok():
            break
        scale = 10.0 ** log_scale
        for direction in [1, -1]:
            if not time_ok():
                break
            b = x0 + direction * scale * max(1.0, abs(x0))
            try:
                fb = f(b)
            except Exception:
                continue
            if not np.isfinite(fb):
                continue
            if fa is not None and np.isfinite(fa) and fa * fb < 0:
                try:
                    root = brentq(f, a, b, xtol=1e-14, rtol=1e-14, maxiter=100)
                    if np.isfinite(root):
                        return root
                except Exception:
                    pass
    
    # If all else fails, return the best x we found
    return best_x
# EVOLVE-BLOCK-END
