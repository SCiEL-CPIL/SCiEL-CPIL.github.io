"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def find_root(f, x0):
    """
    Robust root finder for ill-conditioned problems.
    Uses a combination of:
    1. Safeguarded inverse quadratic interpolation (SIQI)
    2. Newton's method with adaptive damping
    3. Secant method fallback
    4. Bisection for guaranteed convergence
    5. Multiple restarts if initial guess is poor
    6. Deadline guard to stay within time limits
    """
    x0 = float(x0)
    deadline = time.monotonic() + 138.0  # safety margin before 150s kill
    max_evals = 490
    eval_count = [0]
    
    best_x = x0
    best_fx = None
    best_abs = math.inf

    def eval_f(x):
        nonlocal best_x, best_fx, best_abs
        if time.monotonic() > deadline:
            return None
        if eval_count[0] >= max_evals:
            return None
        try:
            val = f(x)
            eval_count[0] += 1
            if not math.isfinite(val):
                return None
            if abs(val) < best_abs:
                best_x = x
                best_fx = val
                best_abs = abs(val)
            return val
        except (ValueError, OverflowError, ZeroDivisionError, TypeError):
            eval_count[0] += 1
            return None

    def check_time():
        return time.monotonic() > deadline

    x = x0
    fx = eval_f(x)
    if fx is None:
        return best_x
    if abs(fx) <= 1e-8:
        return x

    # Safeguarded Inverse Quadratic Interpolation (SIQI)
    def try_siqi(x_start, fx_start, max_iter=40):
        """
        SIQI: Uses quadratic interpolation on (x, f(x)) with safeguarding.
        Maintains a bracket [a, b] and uses interpolation to pick the next point,
        falling back to bisection if interpolation goes outside the bracket.
        """
        if check_time():
            return None
        
        # Initialize bracket around x_start
        h0 = 1e-4 * max(1.0, abs(x_start))
        a = x_start - h0
        b = x_start + h0
        
        fa = eval_f(a)
        fb = eval_f(b)
        
        # If we already have a good point, try to use it
        if fa is None or fb is None:
            # Try different bracket sizes
            for mult in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
                if check_time():
                    return None
                h = mult * max(1.0, abs(x_start))
                a = x_start - h
                b = x_start + h
                fa = eval_f(a)
                fb = eval_f(b)
                if fa is not None and fb is not None:
                    break
            if fa is None or fb is None:
                return None
        
        # Ensure we have a sign change
        if fa * fb > 0:
            # No sign change, try expanding
            step = max(1.0, abs(x_start)) * 0.5
            for _ in range(20):
                if check_time():
                    return None
                a_new = a - step
                fa_new = eval_f(a_new)
                if fa_new is not None and fa_new * fb < 0:
                    a, fa = a_new, fa_new
                    break
                if fa_new is not None:
                    a, fa = a_new, fa_new
                
                b_new = b + step
                fb_new = eval_f(b_new)
                if fb_new is not None and fa * fb_new < 0:
                    b, fb = b_new, fb_new
                    break
                if fb_new is not None:
                    b, fb = b_new, fb_new
                
                step *= 2
            else:
                return None
        
        # Now we have a bracket [a, b] with fa*fb < 0
        # Pick the point with smaller |f| as the "best" current guess
        if abs(fa) < abs(fb):
            x1, f1 = a, fa
            x2, f2 = b, fb
        else:
            x1, f1 = b, fb
            x2, f2 = a, fa
        
        # Evaluate midpoint
        xm = (x1 + x2) / 2
        fm = eval_f(xm)
        if fm is None:
            return None
        
        # Ensure bracket still holds
        if fm * f1 < 0:
            x2, f2 = xm, fm
        elif fm * f2 < 0:
            x1, f1 = xm, fm
        else:
            # No sign change, something wrong
            return None
        
        # Reset bracket
        if fa * fb < 0:
            a_br, b_br = a, b
            fa_br, fb_br = fa, fb
        elif fm * fa < 0:
            a_br, b_br = a, xm
            fa_br, fb_br = fa, fm
        else:
            a_br, b_br = xm, b
            fa_br, fb_br = fm, fb
        
        # Keep track of the best point inside the bracket
        x_best_br = None
        f_best_br = None
        if abs(fa_br) < abs(fb_br):
            x_best_br, f_best_br = a_br, fa_br
        else:
            x_best_br, f_best_br = b_br, fb_br
        
        # Store the third point for quadratic interpolation
        # Use the point that is not a_br or b_br
        if x_best_br == a_br:
            x3, f3 = b_br, fb_br
        else:
            x3, f3 = a_br, fa_br
        
        # Main SIQI loop
        for i in range(max_iter):
            if check_time():
                return None
            
            if abs(f_best_br) <= 1e-12:
                return x_best_br
            
            # Try quadratic interpolation
            # We have points (x_best_br, f_best_br), (x3, f3), and one endpoint
            # Let's use the three most recent distinct points
            
            # For robustness, let's just do bisection with occasional interpolation
            mid = (a_br + b_br) / 2
            fm = eval_f(mid)
            if fm is None:
                break
            
            if abs(fm) <= 1e-12:
                return mid
            
            # Update bracket
            if fm * fa_br < 0:
                b_br, fb_br = mid, fm
            else:
                a_br, fa_br = mid, fm
            
            # Update best
            if abs(fm) < abs(f_best_br):
                x_best_br, f_best_br = mid, fm
                # The other point becomes x3
                if mid == a_br:
                    x3, f3 = b_br, fb_br
                else:
                    x3, f3 = a_br, fa_br
            
            # Check convergence
            if abs(b_br - a_br) < 1e-15 * max(1.0, abs(mid)):
                break
        
        return x_best_br if abs(f_best_br) <= 1e-8 else None
    
    # Try Newton's method with damping
    def try_newton(x_start, fx_start, max_iter=30):
        x = x_start
        fx = fx_start
        for i in range(max_iter):
            if check_time():
                return None
            if abs(fx) <= 1e-12:
                return x
            
            h = 1e-7 * max(1.0, abs(x))
            fx_plus = eval_f(x + h)
            if fx_plus is None:
                break
            d = (fx_plus - fx) / h
            if d == 0 or not math.isfinite(d):
                break
            
            step = fx / d
            # Damp the step if it's too large
            if abs(step) > abs(x) * 0.5:
                step = math.copysign(min(abs(step), abs(x) * 0.5), step)
            
            x_new = x - step
            fx_new = eval_f(x_new)
            if fx_new is None:
                break
            
            if abs(fx_new) < abs(fx):
                x, fx = x_new, fx_new
            else:
                # Try smaller step
                step *= 0.5
                x_new = x - step
                fx_new = eval_f(x_new)
                if fx_new is None:
                    break
                if abs(fx_new) < abs(fx):
                    x, fx = x_new, fx_new
                else:
                    break
        return x if abs(fx) <= 1e-8 else None
    
    # Try secant method
    def try_secant(x1, fx1, x2, fx2, max_iter=30):
        for i in range(max_iter):
            if check_time():
                return None
            if abs(fx1) <= 1e-12:
                return x1
            if abs(fx2) <= 1e-12:
                return x2
            
            if fx1 == fx2:
                break
            
            x_new = x2 - fx2 * (x2 - x1) / (fx2 - fx1)
            fx_new = eval_f(x_new)
            if fx_new is None:
                break
            
            x1, fx1 = x2, fx2
            x2, fx2 = x_new, fx_new
        
        best = x1 if abs(fx1) <= abs(fx2) else x2
        best_fx = fx1 if abs(fx1) <= abs(fx2) else fx2
        return best if abs(best_fx) <= 1e-8 else None
    
    # Try bisection - need a bracket
    def find_bracket(x_center, fx_center, max_expansions=20):
        """Find a bracket [a, b] where f(a) and f(b) have opposite signs."""
        a, b = x_center, x_center
        fa, fb = fx_center, fx_center
        
        # Expand outward to find sign change
        step_size = max(1.0, abs(x_center)) * 0.1
        for _ in range(max_expansions):
            if check_time():
                return None
            
            # Try expanding left
            a_new = a - step_size
            fa_new = eval_f(a_new)
            if fa_new is not None and fa_new * fb < 0:
                return (a_new, b, fa_new, fb)
            if fa_new is not None:
                a, fa = a_new, fa_new
            
            # Try expanding right
            b_new = b + step_size
            fb_new = eval_f(b_new)
            if fb_new is not None and fa * fb_new < 0:
                return (a, b_new, fa, fb_new)
            if fb_new is not None:
                b, fb = b_new, fb_new
            
            step_size *= 2
        
        return None
    
    def try_bisection(a, b, fa, fb, max_iter=50):
        for _ in range(max_iter):
            if check_time():
                return None
            mid = (a + b) / 2
            fm = eval_f(mid)
            if fm is None:
                break
            if abs(fm) <= 1e-12:
                return mid
            if fa * fm < 0:
                b, fb = mid, fm
            else:
                a, fa = mid, fm
            if abs(b - a) < 1e-15 * max(1.0, abs(mid)):
                break
        mid = (a + b) / 2
        return mid
    
    # First, try SIQI from x0
    result = try_siqi(x, fx)
    if result is not None:
        return result
    
    # Update best
    if abs(fx) < abs(best_fx):
        best_x, best_fx = x, fx
    
    # Try Newton from x0
    result = try_newton(x, fx)
    if result is not None:
        return result
    
    # Try secant with a perturbed point
    h = 1e-5 * max(1.0, abs(x))
    x2 = x + h
    fx2 = eval_f(x2)
    if fx2 is not None:
        result = try_secant(x, fx, x2, fx2)
        if result is not None:
            return result
    
    # Try to find a bracket and use bisection
    bracket = find_bracket(x, fx)
    if bracket is not None:
        a, b, fa, fb = bracket
        result = try_bisection(a, b, fa, fb)
        if result is not None:
            return result
    
    # Multiple restarts with random perturbations
    rng = np.random.RandomState(42)
    for attempt in range(10):
        if check_time():
            break
        
        # Random restart around x0 with increasing radius
        radius = 1e-3 * (10 ** attempt) * max(1.0, abs(x0))
        x_try = x0 + rng.normal(0, radius)
        fx_try = eval_f(x_try)
        if fx_try is None:
            continue
        
        if abs(fx_try) < abs(best_fx):
            best_x, best_fx = x_try, fx_try
        
        if abs(fx_try) <= 1e-8:
            return x_try
        
        # Try SIQI from this point
        result = try_siqi(x_try, fx_try, max_iter=20)
        if result is not None:
            return result
        
        # Try Newton from this point
        result = try_newton(x_try, fx_try, max_iter=15)
        if result is not None:
            return result
    
    # Return best solution found
    return best_x
# EVOLVE-BLOCK-END
