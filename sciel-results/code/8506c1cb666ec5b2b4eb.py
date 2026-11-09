"""Root finder for smooth polynomial equations with call-budget awareness."""
import time
import math

_DEADLINE_START = time.time()
_WALL_LIMIT = 115.0  # safety margin before 120s kill
_SAFETY_MARGIN = 5.0


def _time_left():
    return _WALL_LIMIT - _SAFETY_MARGIN - (time.time() - _DEADLINE_START)


def _is_finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def _iqi_root(x0, f0, x1, f1, x2, f2):
    """Inverse quadratic interpolation root from three points."""
    # Lagrange form for inverse quadratic interpolation
    # x = x0 + (f - f0) * [ (x1-x0)(x2-x0) / ((f1-f0)(f2-f0)) ] + ... 
    # Simpler: use the standard IQUAD formula
    denom = (f0 - f1) * (f0 - f2) * (f1 - f2)
    if abs(denom) < 1e-30:
        return None
    
    # x_root = x0*f1*f2*(x1-x0)*(x2-x0)/((f0-f1)*(f0-f2)) + ... 
    # Actually, let's use a cleaner formulation
    # From three points (x0,f0), (x1,f1), (x2,f2), find x where f=0
    # Using Lagrange interpolation on x as function of f:
    # x(f) = x0 * (f-f1)(f-f2)/((f0-f1)(f0-f2)) + x1*(f-f0)(f-f2)/((f1-f0)(f1-f2)) + x2*(f-f0)(f-f1)/((f2-f0)(f2-f1))
    # At f=0:
    term0 = x0 * (f1 * f2) / ((f0 - f1) * (f0 - f2))
    term1 = x1 * (f0 * f2) / ((f1 - f0) * (f1 - f2))
    term2 = x2 * (f0 * f1) / ((f2 - f0) * (f2 - f1))
    x_root = term0 + term1 + term2
    return x_root


def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using:
    1. Secant/IQI hybrid for fast convergence
    2. Bisection fallback for guaranteed convergence
    3. Final polish pass
    
    Returns the best x found (minimizing |f(x)|) within the call budget.
    """
    tol = 1e-8
    max_total_calls = 490  # stay well under 500
    
    def safe_eval(x):
        """Evaluate f(x) safely, returning (fx, ok)."""
        try:
            fx = f(x)
        except Exception:
            return None, False
        if not _is_finite(fx):
            return None, False
        return fx, True
    
    # Track all evaluations to count calls
    call_count = 0
    
    def counted_eval(x):
        nonlocal call_count
        call_count += 1
        return safe_eval(x)
    
    # First, evaluate at x0
    x = float(x0)
    fx, ok = counted_eval(x)
    if not ok:
        # Try nearby points
        for dx in [1e-4, -1e-4, 1e-3, -1e-3, 1e-2, -1e-2, 0.1, -0.1, 1.0, -1.0]:
            x_try = x + dx
            fx, ok = counted_eval(x_try)
            if ok:
                x = x_try
                break
        if not ok:
            return x  # return best effort
    
    if abs(fx) <= tol:
        return x
    
    # Strategy 1: Secant method with IQI when 3 points available
    x_prev = x
    fx_prev = fx
    h = 1e-4 * max(1.0, abs(x))
    x_curr = x + h
    fx_curr, ok = counted_eval(x_curr)
    
    if not ok:
        x_curr = x - h
        fx_curr, ok = counted_eval(x_curr)
    
    best_x = x
    best_fx = fx
    
    # Keep history for IQI
    history = [(x, fx)]
    if ok:
        history.append((x_curr, fx_curr))
    
    # Secant/IQI iterations
    secant_iters = 0
    max_secant_iters = 20
    while secant_iters < max_secant_iters and call_count < max_total_calls:
        if _time_left() < 0:
            break
        if fx_curr is None or fx_prev is None:
            break
        if abs(fx_curr) <= tol:
            return x_curr
        if abs(fx_prev) <= tol:
            return x_prev
        
        # Try IQI if we have 3 distinct points with distinct f values
        if len(history) >= 3:
            (xa, fa), (xb, fb), (xc, fc) = history[-3], history[-2], history[-1]
            if (abs(fa - fb) > 1e-15 and abs(fa - fc) > 1e-15 and abs(fb - fc) > 1e-15):
                x_iqi = _iqi_root(xa, fa, xb, fb, xc, fc)
                if x_iqi is not None and _is_finite(x_iqi):
                    # Use IQI step, but safeguard with secant if IQI goes too far
                    x_secant = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev) if abs(fx_curr - fx_prev) > 1e-15 else x_curr
                    
                    # Choose IQI if it's closer to x_curr than secant, or if secant is bad
                    if abs(x_iqi - x_curr) < abs(x_secant - x_curr) * 2.0:
                        x_next = x_iqi
                    else:
                        x_next = x_secant
                else:
                    x_next = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev) if abs(fx_curr - fx_prev) > 1e-15 else x_curr
            else:
                x_next = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev) if abs(fx_curr - fx_prev) > 1e-15 else x_curr
        else:
            x_next = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev) if abs(fx_curr - fx_prev) > 1e-15 else x_curr
        
        if not _is_finite(x_next):
            break
        
        fx_next, ok = counted_eval(x_next)
        if not ok:
            break
        
        if abs(fx_next) < abs(best_fx):
            best_fx = fx_next
            best_x = x_next
        
        # Update history
        history.append((x_next, fx_next))
        if len(history) > 3:
            history.pop(0)
        
        # Update
        x_prev = x_curr
        fx_prev = fx_curr
        x_curr = x_next
        fx_curr = fx_next
        secant_iters += 1
    
    if abs(best_fx) <= tol:
        return best_x
    
    # Strategy 2: Bisection - need to find a bracket
    # Try to find a sign change around best_x
    x_lo = best_x
    x_hi = best_x
    f_lo = best_fx
    
    # Expand search range to find sign change
    step = 1e-2
    found_bracket = False
    for _ in range(50):
        if call_count >= max_total_calls:
            break
        if _time_left() < 0:
            break
        
        # Try expanding in both directions from best_x
        x_try_left = best_x - step
        f_try_left, ok_l = counted_eval(x_try_left)
        if ok_l and best_fx * f_try_left < 0:
            x_lo = x_try_left
            x_hi = best_x
            f_lo = f_try_left
            f_hi = best_fx
            found_bracket = True
            break
        
        x_try_right = best_x + step
        f_try_right, ok_r = counted_eval(x_try_right)
        if ok_r and best_fx * f_try_right < 0:
            x_lo = best_x
            x_hi = x_try_right
            f_lo = best_fx
            f_hi = f_try_right
            found_bracket = True
            break
        
        step *= 2.0
    
    if found_bracket:
        # Bisection with IQI acceleration
        for _ in range(60):
            if call_count >= max_total_calls:
                break
            if _time_left() < 0:
                break
            
            x_mid = (x_lo + x_hi) / 2.0
            f_mid, ok = counted_eval(x_mid)
            if not ok:
                break
            
            if abs(f_mid) <= tol:
                return x_mid
            
            if abs(f_mid) < abs(best_fx):
                best_fx = f_mid
                best_x = x_mid
            
            if f_lo * f_mid <= 0:
                x_hi = x_mid
                f_hi = f_mid
            else:
                x_lo = x_mid
                f_lo = f_mid
    
    # Strategy 3: Final polish pass — if we're close, use Newton with tighter tol
    if abs(best_fx) < 1e-4 and call_count < max_total_calls and _time_left() > 0:
        x_polish = best_x
        for _ in range(15):
            if call_count >= max_total_calls:
                break
            if _time_left() < 0:
                break
            
            fx_p, ok = counted_eval(x_polish)
            if not ok:
                break
            
            if abs(fx_p) <= tol:
                return x_polish
            
            h = 1e-8 * max(1.0, abs(x_polish))
            fx_p_plus, ok1 = counted_eval(x_polish + h)
            if not ok1:
                break
            fx_p_minus, ok2 = counted_eval(x_polish - h)
            if not ok2:
                d_p = (fx_p_plus - fx_p) / h
            else:
                d_p = (fx_p_plus - fx_p_minus) / (2.0 * h)
            
            if abs(d_p) < 1e-15:
                break
            
            step_p = fx_p / d_p
            x_polish = x_polish - step_p
            
            if not _is_finite(x_polish):
                break
            
            if abs(fx_p) < abs(best_fx):
                best_fx = fx_p
                best_x = x_polish
            
            if abs(step_p) < 1e-14 * max(1.0, abs(x_polish)):
                break
    
    return best_x
