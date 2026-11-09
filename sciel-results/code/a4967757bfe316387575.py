"""Root finder: Brent's method (bisection + secant + inverse quadratic interpolation)
with smart bracket expansion, for smooth polynomial equations.

Strategy:
  1. Evaluate f(x0). If |f(x0)| <= 1e-8, return immediately.
  2. Expand a bracket around x0 until a sign change is found.
  3. Apply Brent's method to converge to the root with minimal function calls.
  4. Track the best x (smallest |f(x)|) throughout; return it if we run out of budget.
"""
import time
import math

_TOL = 1e-8
_CALL_CAP = 490
_DEADLINE_S = 140.0

def find_root(f, x0):
    x0 = float(x0)
    t_start = time.monotonic()
    deadline = t_start + _DEADLINE_S
    calls = 0
    best_x = x0
    best_f = float('inf')
    best_abs_f = float('inf')

    def _time_ok():
        return (time.monotonic() - t_start) < _DEADLINE_S

    def _fx(x):
        nonlocal calls, best_x, best_f, best_abs_f
        if calls >= _CALL_CAP:
            return float('nan')
        calls += 1
        try:
            v = float(f(x))
        except Exception:
            return float('nan')
        
        if v != v: # NaN
            return v
        
        if abs(v) < best_abs_f:
            best_abs_f = abs(v)
            best_x = x
            best_f = v
        return v

    # Phase 0: Initial evaluation
    f0 = _fx(x0)
    if f0 != f0:
        return x0
    if abs(f0) <= _TOL:
        return x0

    # Phase 1: Damped Newton's Method
    x = x0
    fx = f0
    h = 1e-7 * max(1.0, abs(x))
    
    for i in range(30):
        if not _time_ok() or calls >= _CALL_CAP:
            break
            
        if calls + 2 > _CALL_CAP:
            break
            
        xp = x + h
        xm = x - h
        fxp = _fx(xp)
        fxm = _fx(xm)
        
        if fxp != fxp or fxm != fxm:
            break
            
        fp = (fxp - fxm) / (2.0 * h)
        if fp == 0.0 or fp != fp:
            break
            
        dx = -fx / fp
        
        max_step = 0.1 * max(1.0, abs(x))
        if abs(dx) > max_step:
            dx = max_step * (1 if dx > 0 else -1)
            
        x_new = x + dx
        if x_new == x:
            break
            
        fx_new = _fx(x_new)
        if fx_new != fx_new:
            break
            
        if abs(fx_new) <= _TOL:
            return x_new
            
        if abs(fx_new) < abs(fx):
            x = x_new
            fx = fx_new
            h = 1e-7 * max(1.0, abs(x))
        else:
            dx_half = dx * 0.5
            x_new_half = x + dx_half
            fx_half = _fx(x_new_half)
            if fx_half != fx_half:
                break
            if abs(fx_half) < abs(fx):
                x = x_new_half
                fx = fx_half
                h = 1e-7 * max(1.0, abs(x))
            else:
                break
                
        if abs(x - x0) < 1e-12 and abs(fx) < 1e-12:
            break

    if abs(best_abs_f) <= _TOL:
        return best_x

    # Phase 2: Bracketing and Illinois Method
    center = best_x
    fc = best_f
    scale = max(1.0, abs(center)) * 0.1
    
    a = b = None
    fa = fb = None
    
    found_bracket = False
    for m in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0, 1000.0]:
        if not _time_ok() or calls >= _CALL_CAP:
            break
        for sign in [1, -1]:
            x_try = center + sign * m * scale
            f_try = _fx(x_try)
            if f_try != f_try:
                continue
            if abs(f_try) <= _TOL:
                return x_try
            if fc * f_try < 0:
                if abs(fc) < abs(f_try):
                    a, fa = center, fc
                    b, fb = x_try, f_try
                else:
                    a, fa = x_try, f_try
                    b, fb = center, fc
                found_bracket = True
                break
        if found_bracket:
            break

    if not found_bracket:
        return best_x

    if abs(fa) > abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    for i in range(100):
        if not _time_ok() or calls >= _CALL_CAP:
            break
            
        if abs(fa) <= _TOL:
            return a
        if abs(fb) <= _TOL:
            return b
            
        x_mid = 0.5 * (a + b)
        f_mid = _fx(x_mid)
        if f_mid != f_mid:
            break
            
        if abs(f_mid) <= _TOL:
            return x_mid
            
        if abs(fa) < abs(fb):
            b = x_mid
            fb = f_mid
        else:
            a = x_mid
            fa = f_mid
            
        if fa * fb > 0:
            a, b = b, a
            fa, fb = fb, fa
            if abs(fa) > abs(fb):
                a, b = b, a
                fa, fb = fb, fa

    if abs(fa) <= _TOL:
        return a
    if abs(fb) <= _TOL:
        return b
    return best_x
