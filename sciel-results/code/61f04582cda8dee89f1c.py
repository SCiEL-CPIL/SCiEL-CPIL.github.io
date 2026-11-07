import math
import time

def find_root(f, x0):
    """
    Robust scalar root finder for smooth polynomials.
    Strategy:
    1. Secant-based bracketing: Use local slope to predict root location and expand
       bracket intelligently. This is much faster than blind exponential expansion
       for smooth functions.
    2. Brent's Method: Once bracketed, use Inverse Quadratic Interpolation with
       Bisection safeguard for superlinear convergence to high precision.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    TOL = 1e-9
    MAX_CALLS = 480
    
    x = float(x0)
    call_count = 0
    cache = {}

    def _elapsed():
        return time.time() - start_time

    def _time_ok():
        return _elapsed() < deadline and call_count < MAX_CALLS

    def _f(x_val):
        """Cached function evaluation with call counting."""
        nonlocal call_count
        if call_count >= MAX_CALLS:
            return None
        key = x_val
        if key in cache:
            return cache[key]
        try:
            val = f(key)
            call_count += 1
            if not math.isfinite(val):
                cache[key] = None
                return None
            cache[key] = val
            return val
        except Exception:
            call_count += 1
            return None

    # Phase 0: Check if x0 is already a root
    fx0 = _f(x)
    if fx0 is None:
        return x
    if abs(fx0) <= TOL:
        return x

    # Phase 1: Secant-based Bracketing
    # We want to find a, b such that f(a)*f(b) < 0.
    # Start with x0 and a small perturbation to establish a direction.
    # Then use secant steps to jump towards the root.
    
    # Initial step size: relative to x0 or absolute if x0 is small
    h = 1e-4 * max(1.0, abs(x0))
    
    # Try to get a second point to estimate slope
    x_prev = x - h
    f_prev = _f(x_prev)
    
    if f_prev is None:
        # Fallback if initial eval fails
        x_prev = x + h
        f_prev = _f(x_prev)
        
    if f_prev is None:
        return x

    # Check if we already bracketed
    if fx0 * f_prev < 0:
        a, b = x_prev, x
        fa, fb = f_prev, fx0
    else:
        # Both on same side. Use secant method to predict next point.
        # We maintain a "current" point (x_curr, f_curr) and "previous" (x_prev, f_prev)
        # We want to find a point that crosses zero.
        
        x_curr = x
        f_curr = fx0
        
        # If slope is zero, we might be at a local extremum or flat.
        # If f_curr is close to 0, we handled it.
        # If slope is very small, secant step might be huge.
        
        bracket_found = False
        # We also track the "best" point (closest to 0) in case we fail to bracket
        best_x = x_curr if abs(f_curr) < abs(f_prev) else x_prev
        best_f = f_curr if abs(f_curr) < abs(f_prev) else f_prev
        
        for i in range(100): # Max iterations for bracketing
            if not _time_ok():
                break
                
            # Secant step: x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
            denom = f_curr - f_prev
            if abs(denom) < 1e-15:
                # Slope is zero. Try a different direction.
                # Perturb x_curr
                x_new = x_curr + 1.0 * max(1.0, abs(x_curr))
            else:
                x_new = x_curr - f_curr * (x_curr - x_prev) / denom
            
            if not math.isfinite(x_new):
                # Step went to infinity. Try bisection-like expansion from best_x
                x_new = best_x + 1.0 * max(1.0, abs(best_x))
                
            f_new = _f(x_new)
            if f_new is None:
                break
            
            if abs(f_new) <= TOL:
                return x_new
            
            # Update best
            if abs(f_new) < abs(best_f):
                best_f = f_new
                best_x = x_new
                
            # Check for bracket between x_new and x_curr
            if f_new * f_curr < 0:
                a, b = x_new, x_curr
                fa, fb = f_new, f_curr
                bracket_found = True
                break
            # Check for bracket between x_new and x_prev (in case we jumped over)
            elif f_new * f_prev < 0:
                a, b = x_new, x_prev
                fa, fb = f_new, f_prev
                bracket_found = True
                break
            else:
                # No bracket. Move x_prev to x_curr, x_curr to x_new
                x_prev, f_prev = x_curr, f_curr
                x_curr, f_curr = x_new, f_new
                
                # Safety: If we are wandering too far, force a larger step
                # This handles cases where secant steps are oscillating or too small
                if i % 10 == 9:
                    # Try a large step from best_x
                    x_try = best_x + 10.0 * max(1.0, abs(best_x))
                    f_try = _f(x_try)
                    if f_try is not None and f_try * f_curr < 0:
                        a, b = x_try, x_curr
                        fa, fb = f_try, f_curr
                        bracket_found = True
                        break
                    # Also try negative direction
                    x_try2 = best_x - 10.0 * max(1.0, abs(best_x))
                    f_try2 = _f(x_try2)
                    if f_try2 is not None and f_try2 * f_curr < 0:
                        a, b = x_try2, x_curr
                        fa, fb = f_try2, f_curr
                        bracket_found = True
                        break

    if not bracket_found:
        # Fallback: Use the best point found and try secant from there
        # Or just return best point if it's good enough (though TOL is strict)
        # For smooth polynomials, if we haven't bracketed, we might be stuck.
        # Let's try a simple grid search around best_x as a last resort
        # But first, if best_f is very small, return it.
        if abs(best_f) <= TOL:
            return best_x
            
        # Last resort: Secant iteration from best_x without strict bracketing
        # (Risky, but better than failing)
        x_curr = best_x
        f_curr = best_f
        x_prev = x_curr + 1e-4 * max(1.0, abs(x_curr))
        f_prev = _f(x_prev)
        
        if f_prev is None:
            return best_x
            
        for _ in range(50):
            if not _time_ok():
                break
            if abs(f_curr) <= TOL:
                return x_curr
            if abs(f_prev) <= TOL:
                return x_prev
            denom = f_curr - f_prev
            if abs(denom) < 1e-15:
                break
            x_new = x_curr - f_curr * (x_curr - x_prev) / denom
            if not math.isfinite(x_new):
                break
            x_prev, f_prev = x_curr, f_curr
            x_curr, f_new = x_new, _f(x_new)
            if f_new is None:
                break
            if abs(f_new) <= TOL:
                return x_new
        return x_curr

    # Phase 2: Brent's Method on [a, b]
    # Ensure fa < 0 < fb
    if fa > 0:
        a, b = b, a
        fa, fb = fb, fa

    x_a, x_b = a, b
    f_a, f_b = fa, fb
    
    # Initialize history for Inverse Quadratic Interpolation
    # We have 2 points (a, b). We need 3 for IQU.
    # We can use the midpoint or the previous point from bracketing if distinct.
    # Let's initialize c as a (so IQU will fail initially and use secant/bisection)
    x_c = x_a
    f_c = f_a
    
    e = 0.0
    s = 0.0
    
    for _ in range(80):
        if not _time_ok():
            break

        if abs(f_a) <= TOL:
            return x_a
        if abs(f_b) <= TOL:
            return x_b

        # Inverse Quadratic Interpolation
        # Points: (x_a, f_a), (x_c, f_c), (x_b, f_b)
        # Note: In standard Brent, c is the previous b, and a, b are current bracket.
        # Here we maintain a, b as bracket, c as the other point.
        
        denom1 = (f_a - f_c) * (f_a - f_b)
        denom2 = (f_c - f_a) * (f_c - f_b)
        denom3 = (f_b - f_a) * (f_b - f_c)

        x_new = None
        # Check for distinct points to avoid division by zero or singular matrix
        if abs(x_a - x_c) > 1e-15 and abs(x_a - x_b) > 1e-15 and abs(x_c - x_b) > 1e-15:
            if abs(denom1) > 1e-30 and abs(denom2) > 1e-30 and abs(denom3) > 1e-30:
                try:
                    x_new = (x_a * f_c * f_b / denom1
                             + x_c * f_a * f_b / denom2
                             + x_b * f_a * f_c / denom3)
                    if not math.isfinite(x_new):
                        x_new = None
                except (ZeroDivisionError, OverflowError):
                    x_new = None

        # Safeguard: If IQU fails or is outside bracket, use Secant or Bisection
        if x_new is None or not (x_a < x_new < x_b):
            # Try Secant using a and b
            if abs(f_a - f_b) > 1e-15:
                x_sec = x_a - f_a * (x_a - x_b) / (f_a - f_b)
                if x_a < x_sec < x_b:
                    x_new = x_sec
                else:
                    # Bisection
                    x_new = (x_a + x_b) / 2.0
            else:
                # Bisection
                x_new = (x_a + x_b) / 2.0

        f_new = _f(x_new)
        if f_new is None:
            break
        if abs(f_new) <= TOL:
            return x_new

        # Update bracket
        if f_a * f_new < 0:
            # Root in [x_a, x_new]
            x_c, f_c = x_b, f_b
            x_b, f_b = x_new, f_new
        else:
            # Root in [x_new, x_b]
            x_c, f_c = x_a, f_a
            x_a, f_a = x_new, f_new

    # Return the endpoint with smaller |f|
    if abs(f_a) <= abs(f_b):
        return x_a
    else:
        return x_b
