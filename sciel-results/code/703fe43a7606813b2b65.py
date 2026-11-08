"""Root finder for smooth polynomial equations with call-budget awareness."""
import time
import math

_DEADLINE_START = time.time()
_WALL_LIMIT = 115.0  # safety margin before 150s kill
_SAFETY_MARGIN = 5.0


def _time_left():
    return _WALL_LIMIT - _SAFETY_MARGIN - (time.time() - _DEADLINE_START)


def _is_finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using a combination of:
    1. Secant method (derivative-free, robust)
    2. Newton's method with finite-difference derivative (fast convergence)
    3. Bisection fallback for guaranteed convergence
    4. Multiple restarts with different initial points if needed
    
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
    
    # Strategy 1: Secant method (uses 2 points, no derivative needed)
    # Initialize second point
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
    
    # Secant iterations
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
        
        if abs(fx_curr - fx_prev) < 1e-15:
            break
        
        x_next = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)
        
        if not _is_finite(x_next):
            break
        
        fx_next, ok = counted_eval(x_next)
        if not ok:
            break
        
        if abs(fx_next) < abs(best_fx):
            best_fx = fx_next
            best_x = x_next
        
        # Update
        x_prev = x_curr
        fx_prev = fx_curr
        x_curr = x_next
        fx_curr = fx_next
        secant_iters += 1
    
    if abs(best_fx) <= tol:
        return best_x
    
    # Strategy 2: Newton's method with finite-difference derivative
    x_newton = best_x
    for _ in range(20):
        if call_count >= max_total_calls:
            break
        if _time_left() < 0:
            break
        
        fx, ok = counted_eval(x_newton)
        if not ok:
            break
        
        if abs(fx) <= tol:
            return x_newton
        
        h = 1e-6 * max(1.0, abs(x_newton))
        fx_plus, ok1 = counted_eval(x_newton + h)
        if not ok1:
            break
        
        if abs(fx_plus - fx) < 1e-15:
            break
        
        d = (fx_plus - fx) / h
        if abs(d) < 1e-15:
            break
        
        x_newton = x_newton - fx / d
        
        if not _is_finite(x_newton):
            break
        
        if abs(fx) < abs(best_fx):
            best_fx = fx
            best_x = x_newton
    
    if abs(best_fx) <= tol:
        return best_x
    
    # Strategy 3: Bisection - need to find a bracket
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
        
        # Try expanding in both directions
        x_try = x_lo - step
        f_try, ok = counted_eval(x_try)
        if ok and f_lo * f_try < 0:
            x_hi = x_lo
            x_lo = x_try
            f_hi = f_lo
            f_lo = f_try
            found_bracket = True
            break
        
        x_try = x_hi + step
        f_try, ok = counted_eval(x_try)
        if ok and f_lo * f_try < 0:
            x_lo = x_hi
            x_hi = x_try
            f_lo = f_hi
            f_hi = f_try
            found_bracket = True
            break
        
        # Also check from best_x
        x_try = best_x - step
        f_try, ok = counted_eval(x_try)
        if ok:
            if best_fx * f_try < 0:
                x_lo = x_try
                x_hi = best_x
                f_lo = f_try
                f_hi = best_fx
                found_bracket = True
                break
        
        x_try = best_x + step
        f_try, ok = counted_eval(x_try)
        if ok:
            if best_fx * f_try < 0:
                x_lo = best_x
                x_hi = x_try
                f_lo = best_fx
                f_hi = f_try
                found_bracket = True
                break
        
        step *= 2.0
    
    if found_bracket:
        # Bisection
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
    
    return best_x
