import numpy as np
from scipy.optimize import brentq
import time

def find_root(f, x0):
    x0 = float(x0)
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit
    
    # Helper to check time limit
    def time_exceeded():
        return (time.time() - start_time) > deadline
    
    # 1. Check initial point
    try:
        f0 = f(x0)
    except Exception:
        return x0
    
    if abs(f0) <= 1e-8:
        return x0
    
    # Keep track of the best point found (min |f(x)|)
    best_x = x0
    best_f = f0
    
    # 2. Dense local sweep: fine-grained scan around x0
    # Try several small steps to catch nearby sign changes
    scale = max(1.0, abs(x0))
    local_steps = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    
    for step in local_steps:
        if time_exceeded():
            break
        h = step * scale
        # Scan in both directions with a few sub-steps
        for k in range(1, 5):
            if time_exceeded():
                break
            x_pos = x0 + k * h
            x_neg = x0 - k * h
            try:
                fp = f(x_pos)
                fm = f(x_neg)
            except Exception:
                continue
            
            if abs(fp) < abs(best_f):
                best_f = fp
                best_x = x_pos
            if abs(fm) < abs(best_f):
                best_f = fm
                best_x = x_neg
            
            # Check sign changes
            if f0 * fp < 0:
                a, b, fa, fb = x0, x_pos, f0, fp
                return _converge(f, a, b, fa, fb, best_x, time_exceeded)
            if fm * f0 < 0:
                a, b, fa, fb = x_neg, x0, fm, f0
                return _converge(f, a, b, fa, fb, best_x, time_exceeded)
            # Check if fp and fm have opposite signs (oscillation skipped x0)
            if fp * fm < 0:
                a, b, fa, fb = x_neg, x_pos, fm, fp
                return _converge(f, a, b, fa, fb, best_x, time_exceeded)
    
    # 3. Geometric expansion bracketing
    h = 1e-2 * max(1.0, abs(x0))
    
    for i in range(40):
        if time_exceeded():
            break
        
        x_plus = x0 + h
        x_minus = x0 - h
        
        try:
            fp = f(x_plus)
            fm = f(x_minus)
        except Exception:
            break
        
        # Update best point
        if abs(fp) < abs(best_f):
            best_f = fp
            best_x = x_plus
        if abs(fm) < abs(best_f):
            best_f = fm
            best_x = x_minus
        
        # Check for sign change between x0 and x_plus
        if f0 * fp < 0:
            a, b, fa, fb = x0, x_plus, f0, fp
            return _converge(f, a, b, fa, fb, best_x, time_exceeded)
        
        # Check for sign change between x_minus and x0
        if fm * f0 < 0:
            a, b, fa, fb = x_minus, x0, fm, f0
            return _converge(f, a, b, fa, fb, best_x, time_exceeded)
        
        # Check for sign change between x_minus and x_plus
        if fp * fm < 0:
            a, b, fa, fb = x_minus, x_plus, fm, fp
            return _converge(f, a, b, fa, fb, best_x, time_exceeded)
        
        # Expand step size
        h *= 2.0
        
        if h > 1e15:
            break
    
    # 4. Fallback: multi-scale restart with different offsets
    # Try starting from slightly shifted positions
    for offset in [0.1, -0.1, 0.5, -0.5, 2.0, -2.0]:
        if time_exceeded():
            break
        x_start = x0 + offset * max(1.0, abs(x0))
        try:
            f_start = f(x_start)
        except Exception:
            continue
        
        if abs(f_start) < abs(best_f):
            best_f = f_start
            best_x = x_start
        
        if abs(f_start) <= 1e-8:
            return x_start
        
        # Try bracketing from this new starting point
        h2 = 1e-3 * max(1.0, abs(x_start))
        for j in range(20):
            if time_exceeded():
                break
            xp = x_start + h2
            xn = x_start - h2
            try:
                fpp = f(xp)
                fmn = f(xn)
            except Exception:
                break
            
            if abs(fpp) < abs(best_f):
                best_f = fpp
                best_x = xp
            if abs(fmn) < abs(best_f):
                best_f = fmn
                best_x = xn
            
            if f_start * fpp < 0:
                a, b, fa, fb = x_start, xp, f_start, fpp
                return _converge(f, a, b, fa, fb, best_x, time_exceeded)
            if fmn * f_start < 0:
                a, b, fa, fb = xn, x_start, fmn, f_start
                return _converge(f, a, b, fa, fb, best_x, time_exceeded)
            if fpp * fmn < 0:
                a, b, fa, fb = xn, xp, fmn, fpp
                return _converge(f, a, b, fa, fb, best_x, time_exceeded)
            
            h2 *= 2.0
            if h2 > 1e15:
                break
    
    # No bracket found. Return the best point found.
    return best_x


def _converge(f, a, b, fa, fb, best_x, time_exceeded):
    """Converge to root using Brent's method with bisection fallback."""
    try:
        root = brentq(f, a, b, args=(), xtol=1e-14, rtol=1e-14, maxiter=100)
        if abs(f(root)) <= 1e-8:
            return root
        else:
            return best_x
    except Exception:
        # Bisection fallback
        try:
            left, right = a, b
            fl, fr = fa, fb
            for _ in range(100):
                if time_exceeded():
                    break
                mid = (left + right) / 2.0
                fm = f(mid)
                if abs(fm) <= 1e-8:
                    return mid
                if fl * fm < 0:
                    right, fr = mid, fm
                else:
                    left, fl = mid, fm
            return (left + right) / 2.0
        except Exception:
            return best_x
