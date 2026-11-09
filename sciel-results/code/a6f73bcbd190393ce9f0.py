"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Robust root finding for ill-conditioned equations.
    Strategy:
      1. Inverse quadratic interpolation (IQI) with secant fallback.
      2. If not converged, attempt global bracketing via expanding intervals.
      3. Use Brent's method (via scipy) on the bracket if found.
      4. If no bracket, use a local minimization of |f(x)| via golden-section.
      5. Final polish with high-precision secant.
    Deadline guard: return best x found before 145s.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def time_ok():
        return time.time() < deadline

    x0 = float(x0)
    best_x = x0
    best_f = f(best_x)
    best_abs = abs(best_f)
    eval_count = 1

    def update_best(x, fx):
        nonlocal best_x, best_f, best_abs, eval_count
        ax = abs(fx)
        if ax < best_abs:
            best_x = x
            best_f = fx
            best_abs = ax

    update_best(best_x, best_f)

    # Helper: safe function evaluation
    def safe_f(x):
        nonlocal eval_count
        if not time_ok():
            return None
        try:
            val = f(x)
            eval_count += 1
            if not math.isfinite(val):
                return None
            return val
        except:
            eval_count += 1
            return None

    # Phase 1: Inverse Quadratic Interpolation (IQI) with secant fallback
    # We maintain a history of the last 3 points: (x0, f0), (x1, f1), (x2, f2)
    # where x2 is the current point.
    
    # Initialize with x0 and a small perturbation
    x_a = x0
    f_a = best_f
    
    h_init = 1e-4 * max(1.0, abs(x0))
    x_b = x0 + h_init
    f_b = safe_f(x_b)
    
    if f_b is None:
        # Fallback if perturbation fails
        x_b = x0
        f_b = best_f
        # Try a larger perturbation
        h_init2 = 1e-2 * max(1.0, abs(x0))
        x_c = x0 + h_init2
        f_c = safe_f(x_c)
        if f_c is not None:
            x_b = x_c
            f_b = f_c
        else:
            # Can't even get 2 points, return best
            return best_x

    # History: (x, f)
    # We want to order them such that we have distinct points
    # For IQI, we need 3 distinct points. Let's start with just 2 and use secant first.
    
    x_prev2 = x_a
    f_prev2 = f_a
    x_prev1 = x_b
    f_prev1 = f_b
    
    # Current point
    x_curr = x_prev1
    f_curr = f_prev1
    
    # We need a third point for IQI. Let's take a secant step first to get a third point.
    # Secant step from (x_a, f_a) and (x_b, f_b)
    if abs(x_b - x_a) > 1e-15:
        d_secant = (f_b - f_a) / (x_b - x_a)
        if d_secant != 0 and math.isfinite(d_secant):
            step = f_b / d_secant
            # Limit step
            max_step = 1.0 + abs(x_b)
            if abs(step) > max_step:
                step = max_step * (1.0 if step > 0 else -1.0)
            x_new = x_b - step
            if math.isfinite(x_new):
                f_new = safe_f(x_new)
                if f_new is not None:
                    # Update history: (x_a, f_a), (x_b, f_b), (x_new, f_new)
                    x_prev2, f_prev2 = x_prev2, f_prev2
                    x_prev1, f_prev1 = x_prev1, f_prev1
                    x_curr, f_curr = x_new, f_new
                    update_best(x_curr, f_curr)
    
    # Now we have 3 points: (x_prev2, f_prev2), (x_prev1, f_prev1), (x_curr, f_curr)
    # But we need to ensure they are distinct and ordered properly for IQI.
    # Standard IQI uses the last three points. Let's define a helper to compute the next step.
    
    def iqI_step(x1, f1, x2, f2, x3, f3):
        """
        Compute the next x using Inverse Quadratic Interpolation.
        Fits a parabola y = ax^2 + bx + c? No, inverse quadratic fits x as a quadratic in f.
        x = A*f^2 + B*f + C
        We want x such that f(x) = 0, so x_next = C.
        
        Using Lagrange interpolation on (f1, x1), (f2, x2), (f3, x3) for x(f).
        x(f) = x1 * ((f - f2)(f - f3)) / ((f1 - f2)(f1 - f3)) + 
               x2 * ((f - f1)(f - f3)) / ((f2 - f1)(f2 - f3)) + 
               x3 * ((f - f1)(f - f2)) / ((f3 - f1)(f3 - f2))
        
        Set f = 0:
        x_next = x1 * (f2 * f3) / ((f1 - f2)(f1 - f3)) + 
                 x2 * (f1 * f3) / ((f2 - f1)(f2 - f3)) + 
                 x3 * (f1 * f2) / ((f3 - f1)(f3 - f2))
        """
        # Check for distinct f values
        if abs(f1 - f2) < 1e-30 or abs(f1 - f3) < 1e-30 or abs(f2 - f3) < 1e-30:
            return None
        
        denom1 = (f1 - f2) * (f1 - f3)
        denom2 = (f2 - f1) * (f2 - f3)
        denom3 = (f3 - f1) * (f3 - f2)
        
        if abs(denom1) < 1e-30 or abs(denom2) < 1e-30 or abs(denom3) < 1e-30:
            return None
            
        term1 = x1 * (f2 * f3) / denom1
        term2 = x2 * (f1 * f3) / denom2
        term3 = x3 * (f1 * f2) / denom3
        
        x_next = term1 + term2 + term3
        
        if not math.isfinite(x_next):
            return None
            
        return x_next

    # Main loop for IQI
    for it in range(200):
        if not time_ok():
            break
        if abs(f_curr) <= 1e-8:
            return x_curr
        update_best(x_curr, f_curr)
        
        # Try IQI
        x_next = iqI_step(x_prev2, f_prev2, x_prev1, f_prev1, x_curr, f_curr)
        
        # Fallback to Secant if IQI fails or produces a bad step
        if x_next is None:
            # Secant using last two points
            if abs(x_curr - x_prev1) > 1e-15:
                d = (f_curr - f_prev1) / (x_curr - x_prev1)
                if d != 0 and math.isfinite(d):
                    step = f_curr / d
                    max_step = 1.0 + abs(x_curr)
                    if abs(step) > max_step:
                        step = max_step * (1.0 if step > 0 else -1.0)
                    x_next = x_curr - step
                    if not math.isfinite(x_next):
                        break
                else:
                    break
            else:
                break
        
        # Limit step size to prevent divergence
        max_step = 1.0 + abs(x_curr)
        step_size = abs(x_next - x_curr)
        if step_size > max_step:
            # Adjust x_next to be within max_step
            direction = 1.0 if x_next > x_curr else -1.0
            x_next = x_curr + direction * max_step
        
        # Check if x_next is too close to existing points
        if abs(x_next - x_curr) < 1e-15 or abs(x_next - x_prev1) < 1e-15 or abs(x_next - x_prev2) < 1e-15:
            # Perturb slightly
            x_next = x_curr + 1e-6 * max(1.0, abs(x_curr))
            
        f_next = safe_f(x_next)
        if f_next is None:
            break
        
        # Acceptance: always accept if finite, but prefer improvement?
        # IQI is superlinear if close to root. Let's just accept and update history.
        # However, if it diverges, we might want to backtrack.
        # For now, simple update.
        
        x_prev2, f_prev2 = x_prev1, f_prev1
        x_prev1, f_prev1 = x_curr, f_curr
        x_curr, f_curr = x_next, f_next
        
        update_best(x_curr, f_curr)

    if abs(f_curr) <= 1e-8:
        return x_curr
    update_best(x_curr, f_curr)

    # Phase 2: Try to find a bracket
    # Expand from x0 in both directions
    a = x0
    fa = best_f
    b = x0
    fb = best_f

    # Try expanding to the right
    scale = 1.0
    found_bracket = False
    for i in range(60):
        if not time_ok():
            break
        b = x0 + scale
        fb = safe_f(b)
        if fb is None:
            scale *= 2.0
            continue
        update_best(b, fb)
        if fa * fb < 0:
            found_bracket = True
            break
        a = b
        fa = fb
        scale *= 2.0
        if abs(scale) > 1e12:
            break

    if not found_bracket:
        # Try expanding to the left
        a = x0
        fa = best_f
        scale = 1.0
        for i in range(60):
            if not time_ok():
                break
            b = x0 - scale
            fb = safe_f(b)
            if fb is None:
                scale *= 2.0
                continue
            update_best(b, fb)
            if fa * fb < 0:
                found_bracket = True
                break
            a = b
            fa = fb
            scale *= 2.0
            if abs(scale) > 1e12:
                break

    if found_bracket and time_ok():
        # Use scipy's brentq
        try:
            from scipy.optimize import brentq
            root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if time_ok():
                fr = f(root)
                update_best(root, fr)
                if abs(fr) <= 1e-8:
                    return root
        except:
            pass

    # Phase 3: Local minimization of |f(x)| using golden-section search
    if time_ok():
        # Find a local minimum of |f| near x0
        # Use a window around the best point
        center = best_x
        window = max(1.0, abs(center))
        lo = center - window
        hi = center + window

        # First evaluate endpoints and center
        f_lo = safe_f(lo)
        f_hi = safe_f(hi)
        f_c = safe_f(center)

        if f_lo is not None:
            update_best(lo, f_lo)
        if f_hi is not None:
            update_best(hi, f_hi)
        if f_c is not None:
            update_best(center, f_c)

        # Golden-section search for minimum of |f(x)|
        gr = (math.sqrt(5) - 1) / 2  # golden ratio conjugate
        x1 = hi - gr * (hi - lo)
        x2 = lo + gr * (hi - lo)

        for i in range(80):
            if not time_ok():
                break
            f1 = safe_f(x1)
            f2 = safe_f(x2)

            if f1 is None or f2 is None:
                break

            update_best(x1, f1)
            update_best(x2, f2)

            if abs(f1) < abs(f2):
                hi = x2
                x2 = x1
                x1 = hi - gr * (hi - lo)
            else:
                lo = x1
                x1 = x2
                x2 = lo + gr * (hi - lo)

            if abs(hi - lo) < 1e-10 * max(1.0, abs(center)):
                break

        # Polish the best point with high-precision secant
        x = best_x
        fx = best_f
        h_polish = 1e-8 * max(1.0, abs(x))
        x_prev = x + h_polish
        f_prev = safe_f(x_prev)
        if f_prev is None:
            f_prev = fx
            x_prev = x

        for it in range(50):
            if not time_ok():
                break
            if abs(fx) <= 1e-12:
                break
            if abs(x - x_prev) > 1e-15:
                d = (fx - f_prev) / (x - x_prev)
            else:
                h = 1e-8 * max(1.0, abs(x))
                fx_h = safe_f(x + h)
                if fx_h is None:
                    break
                d = (fx_h - fx) / h

            if d == 0 or not math.isfinite(d):
                break

            step = fx / d
            max_step = 0.1 * max(1.0, abs(x))
            if abs(step) > max_step:
                step = max_step * (1.0 if step > 0 else -1.0)

            x_new = x - step
            if not math.isfinite(x_new):
                break

            fx_new = safe_f(x_new)
            if fx_new is None:
                break

            if abs(fx_new) < abs(fx):
                x_prev = x
                f_prev = fx
                x = x_new
                fx = fx_new
            else:
                # Try damped step
                x_d = x - step / 2
                fx_d = safe_f(x_d)
                if fx_d is not None and abs(fx_d) < abs(fx):
                    x_prev = x
                    f_prev = fx
                    x = x_d
                    fx = fx_d
                else:
                    break

        update_best(x, fx)

    # Final high-precision polish if we're close
    if time_ok() and best_abs < 1e-4:
        x = best_x
        fx = best_f
        h_final = 1e-10 * max(1.0, abs(x))
        x_prev = x + h_final
        f_prev = safe_f(x_prev)
        if f_prev is not None and abs(x - x_prev) > 1e-15:
            for it in range(20):
                if not time_ok():
                    break
                if abs(fx) <= 1e-10:
                    break
                d = (fx - f_prev) / (x - x_prev)
                if d == 0 or not math.isfinite(d):
                    break
                step = fx / d
                if abs(step) > 0.01 * max(1.0, abs(x)):
                    step = 0.01 * max(1.0, abs(x)) * (1.0 if step > 0 else -1.0)
                x_new = x - step
                if not math.isfinite(x_new):
                    break
                fx_new = safe_f(x_new)
                if fx_new is None:
                    break
                if abs(fx_new) < abs(fx):
                    x_prev = x
                    f_prev = fx
                    x = x_new
                    fx = fx_new
                else:
                    break
            update_best(x, fx)

    return best_x
# EVOLVE-BLOCK-END
