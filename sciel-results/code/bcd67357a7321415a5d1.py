"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Robust scalar root finder.
    Strategy:
    1. Try Newton's method (fast convergence, few calls) with safeguards.
    2. If Newton fails or diverges, try to establish a bracket and use Bisection/Brent.
    3. Always track the best result found.
    4. Return best result if deadline is approached.
    """
    start_time = time.monotonic()
    DEADLINE = start_time + 115.0  # Safety margin before 120s limit
    
    best_x = float(x0)
    best_fx = None
    call_count = 0

    def safe_f(val):
        nonlocal call_count
        try:
            result = f(val)
            call_count += 1
            if result is not None and math.isfinite(result):
                return float(result)
        except Exception:
            pass
        return None

    # Initial evaluation
    best_fx = safe_f(best_x)
    if best_fx is not None and abs(best_fx) <= 1e-8:
        return best_x

    # --- Phase 1: Newton's Method ---
    # Newton is quadratic, so it should solve smooth polynomials in <10 steps
    x_newton = best_x
    fx = best_fx
    newton_success = False
    
    for i in range(25):  # Limit Newton steps to avoid wasting calls
        if time.monotonic() > DEADLINE:
            break
        
        if fx is None:
            break
            
        if abs(fx) <= 1e-10:
            return x_newton

        # Finite difference derivative
        h = 1e-7 * max(1.0, abs(x_newton))
        fx_plus = safe_f(x_newton + h)
        if fx_plus is None:
            break
            
        d = (fx_plus - fx) / h
        if d == 0 or not math.isfinite(d):
            # Derivative is zero or bad, Newton can't proceed
            break
            
        step = fx / d
        
        # Safeguard: Damp step if it's too large
        max_step = 10.0 * max(1.0, abs(x_newton))
        if abs(step) > max_step:
            step = math.copysign(max_step, step)
            
        x_next = x_newton - step
        
        if not math.isfinite(x_next):
            break
            
        fx_next = safe_f(x_next)
        if fx_next is None:
            break
            
        # Update best
        if best_fx is None or abs(fx_next) < abs(best_fx):
            best_x = x_next
            best_fx = fx_next
            
        # Check convergence
        if abs(fx_next) <= 1e-10:
            return x_next
            
        # Check for divergence/stagnation
        if abs(fx_next) > abs(fx) * 1.1:
            # If error is increasing, Newton is likely diverging
            # Break to try bracketing
            break
            
        x_newton = x_next
        fx = fx_next

    # If Newton succeeded in getting very close, return
    if best_fx is not None and abs(best_fx) <= 1e-8:
        return best_x

    # --- Phase 2: Bracketing and Bisection ---
    # If Newton failed, we need a robust method.
    # Try to find a bracket [a, b] where f(a) and f(b) have opposite signs.
    
    scale = max(1.0, abs(x0))
    found_bracket = False
    a = None
    b = None
    fa = None
    fb = None
    
    # Try expanding intervals around x0
    # Start with small expansion to save calls if root is close
    multipliers = [0.1, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
    
    for mult in multipliers:
        if time.monotonic() > DEADLINE:
            break
            
        left = x0 - scale * mult
        right = x0 + scale * mult
        
        fl = safe_f(left)
        fr = safe_f(right)
        
        # Update best
        if fl is not None and (best_fx is None or abs(fl) < abs(best_fx)):
            best_x = left
            best_fx = fl
        if fr is not None and (best_fx is None or abs(fr) < abs(best_fx)):
            best_x = right
            best_fx = fr
            
        if fl is None or fr is None:
            continue
            
        if fl * fr < 0:
            found_bracket = True
            a, b = left, right
            fa, fb = fl, fr
            break
        elif abs(fl) <= 1e-10:
            return left
        elif abs(fr) <= 1e-10:
            return right

    # If no bracket found, try a few more points or just return best
    # Since these are polynomials, a bracket should exist if we expand enough.
    # If we still don't have one, we might be in a region where f doesn't cross 0
    # or we ran out of time/calls.
    
    if found_bracket:
        # Perform Bisection
        for _ in range(60):  # 60 bisections gives ~1e-18 precision
            if time.monotonic() > DEADLINE:
                break
                
            mid = (a + b) / 2.0
            fm = safe_f(mid)
            
            if fm is None:
                break
                
            if abs(fm) <= 1e-10:
                return mid
                
            if (best_fx is None or abs(fm) < abs(best_fx)):
                best_x = mid
                best_fx = fm
                
            if fa * fm < 0:
                b = mid
                fb = fm
            else:
                a = mid
                fa = fm
                
            # Check interval width
            if abs(b - a) < 1e-15:
                break
                
        # Return the better endpoint or mid if we broke out
        if best_fx is not None and abs(best_fx) <= 1e-8:
            return best_x
        # Return the point with smallest |f(x)| from the last bracket
        if fa is not None and fb is not None:
            if abs(fa) < abs(fb):
                return a
            else:
                return b

    # Fallback: Return best found
    return best_x
# EVOLVE-BLOCK-END
