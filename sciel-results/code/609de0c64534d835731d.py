"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def find_root(f, x0):
    """
    Find a root x of f such that |f(x)| <= 1e-8.
    Uses a combination of Newton's method, bisection, and secant method.
    Tracks call count and time to stay within limits.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    max_calls = 480  # Stay well under 500
    
    call_count = 0
    best_x = float(x0)
    best_f = None
    try:
        best_f = f(x0)
        call_count += 1
    except:
        best_f = float('inf')
    
    if best_f is not None and math.isfinite(best_f) and abs(best_f) <= 1e-8:
        return best_x
    
    def _f(x):
        nonlocal call_count
        if call_count >= max_calls:
            raise RuntimeError("call limit")
        if time.time() > deadline:
            raise RuntimeError("time limit")
        call_count += 1
        val = f(x)
        if not math.isfinite(val):
            return float('inf')
        return val
    
    def _update_best(x, fx):
        nonlocal best_x, best_f
        if fx is not None and math.isfinite(fx) and (best_f is None or not math.isfinite(best_f) or abs(fx) < abs(best_f)):
            best_x = x
            best_f = fx
    
    # Try to find a bracket first
    a = float(x0)
    fa = best_f
    _update_best(a, fa)
    
    # Explore nearby points to find a sign change
    h = 1e-4 * max(1.0, abs(a))
    bracket_found = False
    b = None
    fb = None
    
    # Try several offsets
    offsets = [h, -h, 10*h, -10*h, 100*h, -100*h, 1e-2, -1e-2, 1.0, -1.0, 10.0, -10.0]
    for off in offsets:
        if call_count >= max_calls - 10 or time.time() > deadline - 5.0:
            break
        x_try = a + off
        try:
            f_try = _f(x_try)
        except:
            continue
        _update_best(x_try, f_try)
        if math.isfinite(f_try) and math.isfinite(fa) and fa * f_try < 0:
            bracket_found = True
            b = x_try
            fb = f_try
            break
        if abs(f_try) <= 1e-8:
            return x_try
    
    # If no bracket found, use Newton/secant hybrid
    if not bracket_found:
        # Use secant method with two nearby points
        x1 = a
        f1 = fa
        h2 = 1e-5 * max(1.0, abs(a))
        x2 = a + h2
        try:
            f2 = _f(x2)
        except:
            f2 = fa + 1.0  # fallback
        _update_best(x2, f2)
        
        for _ in range(50):
            if call_count >= max_calls - 5 or time.time() > deadline - 5.0:
                break
            if abs(f2) <= 1e-8:
                return x2
            if abs(f1) <= 1e-8:
                return x1
            if f1 == f2:
                break
            # Secant step
            x_new = x2 - f2 * (x2 - x1) / (f2 - f1)
            if not math.isfinite(x_new):
                break
            # Damping if step is too large
            if abs(x_new - x2) > 10 * abs(x2) + 10:
                x_new = x2 + 0.5 * (x2 - x1)
            try:
                f_new = _f(x_new)
            except:
                break
            _update_best(x_new, f_new)
            if abs(f_new) <= 1e-8:
                return x_new
            x1, f1 = x2, f2
            x2, f2 = x_new, f_new
    else:
        # Bisection within bracket [a, b]
        # Ensure fa * fb < 0
        if fa * fb > 0:
            # Swap if needed
            a, b = b, a
            fa, fb = fb, fa
        if fa * fb > 0:
            # No valid bracket, fall back to secant
            pass
        else:
            for _ in range(60):
                if call_count >= max_calls - 5 or time.time() > deadline - 5.0:
                    break
                mid = 0.5 * (a + b)
                if mid == a or mid == b:
                    break
                try:
                    fmid = _f(mid)
                except:
                    break
                _update_best(mid, fmid)
                if abs(fmid) <= 1e-8:
                    return mid
                if fa * fmid <= 0:
                    b = mid
                    fb = fmid
                else:
                    a = mid
                    fa = fmid
    
    # Return best found
    if best_f is not None and math.isfinite(best_f) and abs(best_f) <= 1e-8:
        return best_x
    return best_x
# EVOLVE-BLOCK-END
