import math
import time

_START_TIME = time.time()
_WALL_LIMIT = 115.0  # safety margin before 120s

def _time_left():
    return _WALL_LIMIT - (time.time() - _START_TIME)

def _is_finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)

def find_root(f, x0):
    x = float(x0)
    
    # Check initial point
    try:
        fx = f(x)
    except Exception:
        return x
    if not _is_finite(fx):
        return x
    if abs(fx) <= 1e-8:
        return x
    
    best_x = x
    best_fx = fx
    
    # Newton's method with finite-difference derivative
    for i in range(50):
        if _time_left() < 2.0:
            break
        
        h = 1e-7 * max(1.0, abs(x))
        try:
            fx_plus = f(x + h)
        except Exception:
            break
        
        if not _is_finite(fx_plus):
            break
        
        d = (fx_plus - fx) / h
        if d == 0 or not _is_finite(d):
            break
        
        x_new = x - fx / d
        if not _is_finite(x_new):
            break
        
        try:
            fx_new = f(x_new)
        except Exception:
            break
        
        if not _is_finite(fx_new):
            break
        
        if abs(fx_new) < abs(best_fx):
            best_x = x_new
            best_fx = fx_new
        
        x = x_new
        fx = fx_new
        
        if abs(fx) <= 1e-8:
            return x
    
    return best_x
