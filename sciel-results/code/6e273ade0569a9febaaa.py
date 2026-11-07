import time
import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    start_time = time.time()
    TIME_LIMIT = 140.0  # Safety margin
    
    x0 = float(x0)
    
    # Helper to check time
    def time_ok():
        return (time.time() - start_time) < TIME_LIMIT

    # Initial evaluation
    try:
        f0 = f(x0)
    except Exception:
        return x0
        
    if not np.isfinite(f0):
        # If x0 is bad, try a small offset
        x_try = x0 + 1e-6
        try:
            f_try = f(x_try)
            if np.isfinite(f_try) and abs(f_try) < abs(f0):
                x0 = x_try
                f0 = f_try
        except Exception:
            pass
        if not np.isfinite(f0):
            return x0

    best_x = x0
    best_abs_f = abs(f0)
    
    if best_abs_f <= 1e-8:
        return x0

    # Bracketing Strategy
    # We maintain a left and right bound, expanding outwards.
    # We check for sign changes in [left, x0] and [x0, right].
    
    # Initial small step to catch nearby roots
    step = 0.1
    
    x_left = x0 - step
    x_right = x0 + step
    
    f_left = None
    f_right = None
    
    # Evaluate initial neighbors
    try:
        f_left = f(x_left)
        if np.isfinite(f_left) and abs(f_left) < best_abs_f:
            best_abs_f = abs(f_left)
            best_x = x_left
    except Exception:
        f_left = float('nan')
        
    try:
        f_right = f(x_right)
        if np.isfinite(f_right) and abs(f_right) < best_abs_f:
            best_abs_f = abs(f_right)
            best_x = x_right
    except Exception:
        f_right = float('nan')

    # Check for sign change
    def has_sign_change(fa, fb):
        if not (np.isfinite(fa) and np.isfinite(fb)):
            return False
        return (fa < 0 and fb > 0) or (fa > 0 and fb < 0)

    if has_sign_change(f0, f_left):
        a, b = x_left, x0
        fa, fb = f_left, f0
        # Refine
        try:
            root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=50)
            if np.isfinite(root):
                return root
        except Exception:
            pass
        return best_x

    if has_sign_change(f0, f_right):
        a, b = x0, x_right
        fa, fb = f0, f_right
        try:
            root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=50)
            if np.isfinite(root):
                return root
        except Exception:
            pass
        return best_x

    # Expand search
    # Increase step size. Doubling is standard.
    # We need to update f_left and f_right as we move them.
    # Current state:
    # x_left has f_left, x_right has f_right.
    # We want to move x_left further left and x_right further right.
    
    # To save calls, we can reuse the old boundary as the new inner boundary?
    # No, brentq needs a bracket.
    # If we move x_left to x_new_left, we need f(x_new_left).
    # The segment [x_new_left, x_left] might contain a root.
    # The segment [x_left, x0] was already checked (no sign change).
    # So we just need to check [x_new_left, x_left].
    
    max_expansions = 20
    for i in range(max_expansions):
        if not time_ok():
            break
            
        step *= 2.0
        x_new_left = x_left - step
        x_new_right = x_right + step
        
        f_new_left = float('nan')
        f_new_right = float('nan')
        
        try:
            f_new_left = f(x_new_left)
            if np.isfinite(f_new_left) and abs(f_new_left) < best_abs_f:
                best_abs_f = abs(f_new_left)
                best_x = x_new_left
        except Exception:
            pass
            
        try:
            f_new_right = f(x_new_right)
            if np.isfinite(f_new_right) and abs(f_new_right) < best_abs_f:
                best_abs_f = abs(f_new_right)
                best_x = x_new_right
        except Exception:
            pass

        # Check left expansion: [x_new_left, x_left]
        if has_sign_change(f_new_left, f_left):
            a, b = x_new_left, x_left
            try:
                root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=50)
                if np.isfinite(root):
                    return root
            except Exception:
                pass
            return best_x

        # Check right expansion: [x_right, x_new_right]
        if has_sign_change(f_right, f_new_right):
            a, b = x_right, x_new_right
            try:
                root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=50)
                if np.isfinite(root):
                    return root
            except Exception:
                pass
            return best_x

        # Update boundaries for next iteration
        x_left = x_new_left
        f_left = f_new_left
        x_right = x_new_right
        f_right = f_new_right

    return best_x
