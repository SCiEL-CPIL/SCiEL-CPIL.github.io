"""Root finder: Efficient Newton-secant hybrid for smooth polynomials.

Strategy:
  1. Evaluate f(x0). If |f(x0)| <= 1e-8, return immediately.
  2. Use secant method (2 points) to estimate derivative and take Newton-like steps.
  3. If step is too large, damp it. If no progress, switch to bisection with bracketing.
  4. Track best x throughout; return best if budget exhausted.
"""
import time
import math

_TOL = 1e-8
_CALL_CAP = 490
_DEADLINE_S = 140.0

def find_root(f, x0):
    x0 = float(x0)
    t_start = time.monotonic()
    calls = 0
    best_x = x0
    best_abs_f = float('inf')

    def _time_ok():
        return (time.monotonic() - t_start) < _DEADLINE_S

    def _fx(x):
        nonlocal calls, best_x, best_abs_f
        if calls >= _CALL_CAP:
            return float('nan')
        calls += 1
        try:
            v = float(f(x))
        except Exception:
            return float('nan')
        
        if v != v:  # NaN
            return v
        
        av = abs(v)
        if av < best_abs_f:
            best_abs_f = av
            best_x = x
        return v

    # Phase 0: Initial evaluation
    f0 = _fx(x0)
    if f0 != f0:
        return x0
    if abs(f0) <= _TOL:
        return x0

    # Phase 1: Secant-Newton hybrid
    # Start with two close points for derivative estimation
    h = 1e-6 * max(1.0, abs(x0))
    x1 = x0
    f1 = f0
    x2 = x0 + h
    f2 = _fx(x2)
    
    if f2 != f2:
        return best_x
    if abs(f2) <= _TOL:
        return x2

    # Secant method with damping
    for i in range(25):
        if not _time_ok() or calls >= _CALL_CAP - 1:
            break
        
        # Secant slope
        if x2 == x1:
            break
        slope = (f2 - f1) / (x2 - x1)
        if slope == 0.0 or slope != slope:
            break
        
        # Newton step
        dx = -f1 / slope
        x_new = x1 + dx
        
        # Damp if step is too large
        max_step = 0.5 * max(1.0, abs(x1))
        if abs(dx) > max_step:
            dx = max_step * (1 if dx > 0 else -1)
            x_new = x1 + dx
        
        if x_new == x1:
            break
            
        f_new = _fx(x_new)
        if f_new != f_new:
            break
            
        if abs(f_new) <= _TOL:
            return x_new
            
        # Update points: keep the one with smaller |f|
        if abs(f_new) < abs(f1):
            x1, f1 = x_new, f_new
        else:
            x2, f2 = x_new, f_new
        
        # Check convergence
        if abs(x_new - x1) < 1e-14 and abs(f_new) < 1e-14:
            break

    if abs(best_abs_f) <= _TOL:
        return best_x

    # Phase 2: Bracketing fallback - expand around best_x
    center = best_x
    fc = _fx(center) if best_abs_f > _TOL else 0.0
    if fc != fc:
        fc = float('inf')
    
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
            if fc != float('inf') and fc * f_try < 0:
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

    # Bisection on bracket
    if abs(fa) > abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    for i in range(60):
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
            
        if fa * f_mid < 0:
            b = x_mid
            fb = f_mid
        else:
            a = x_mid
            fa = f_mid

    if abs(fa) <= _TOL:
        return a
    if abs(fb) <= _TOL:
        return b
    return best_x
