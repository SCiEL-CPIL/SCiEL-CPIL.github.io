"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def find_root(f, x0):
    """
    Robust root finder for ill-conditioned problems.
    Uses a combination of:
    1. Newton's method with adaptive damping
    2. Safeguarded inverse quadratic interpolation (SIQI)
    3. Secant method fallback
    4. Multiple restarts with increasing radii
    5. Deadline guard to stay within time limits
    """
    deadline = time.time() + 140  # Leave 10s margin before 150s kill
    max_evals = 480
    eval_count = [0]
    
    def eval_f(x):
        if time.time() > deadline:
            return None
        if eval_count[0] >= max_evals:
            return None
        try:
            val = f(x)
            eval_count[0] += 1
            if not math.isfinite(val):
                return None
            return val
        except (ValueError, OverflowError, ZeroDivisionError, TypeError, FloatingPointError):
            eval_count[0] += 1
            return None
    
    x = float(x0)
    fx = eval_f(x)
    if fx is None:
        return x
    if abs(fx) <= 1e-8:
        return x
    
    best_x = x
    best_fx = fx
    
    def try_newton(x_start, fx_start, max_iter=25):
        """Newton's method with adaptive damping."""
        x = x_start
        fx = fx_start
        for i in range(max_iter):
            if time.time() > deadline:
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
    
    def find_bracket(x_center, fx_center, max_expansions=20):
        """Find a bracket [a, b] where f(a) and f(b) have opposite signs."""
        step_size = max(1e-3, abs(x_center) * 0.01)
        a = x_center - step_size
        b = x_center + step_size
        fa = eval_f(a)
        fb = eval_f(b)
        
        if fa is None or fb is None:
            return None
        
        if fa * fb < 0:
            return (a, b, fa, fb)
        
        # Expand outward to find sign change
        for _ in range(max_expansions):
            if time.time() > deadline:
                return None
            
            a_new = a - step_size
            fa_new = eval_f(a_new)
            if fa_new is not None:
                if fa_new * fb < 0:
                    return (a_new, b, fa_new, fb)
                a, fa = a_new, fa_new
            
            b_new = b + step_size
            fb_new = eval_f(b_new)
            if fb_new is not None:
                if fa * fb_new < 0:
                    return (a, b_new, fa, fb_new)
                b, fb = b_new, fb_new
            
            step_size *= 2
        
        return None
    
    def try_siqi(a, b, fa, fb, max_iter=60):
        """
        Safeguarded Inverse Quadratic Interpolation.
        Maintains a bracket [a, b] and uses quadratic interpolation
        through the last 3 points, falling back to bisection if
        the interpolated point falls outside the bracket.
        """
        if fa is None or fb is None:
            return None
        if fa * fb > 0:
            return None
        
        # Initialize with 3 points for quadratic interpolation
        # Start with a, b, and midpoint
        mid = (a + b) / 2.0
        fm = eval_f(mid)
        if fm is None:
            return None
        
        if abs(fm) <= 1e-8:
            return mid
        
        # Update bracket based on midpoint
        if fm * fa < 0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
        
        # Now we have bracket [a,b] with fa*fb<0
        # Track the 3 most recent points for quadratic interpolation
        # x1 = a, x2 = b, x3 = previous midpoint (or best interior point)
        # We'll maintain a list of recent (x, f) pairs
        
        # For SIQI, we need 3 points. Let's use the bracket endpoints
        # and the point with smaller |f| as the "best" point.
        
        # Store history for interpolation
        # points: list of (x, fx) - keep last 3
        points = [(a, fa), (b, fb)]
        # Add the midpoint that gave us the current bracket
        # Actually, let's just track the best interior point
        if abs(fa) < abs(fb):
            x_best, f_best = a, fa
        else:
            x_best, f_best = b, fb
        
        # Main loop
        for i in range(max_iter):
            if time.time() > deadline:
                break
            
            if abs(f_best) <= 1e-8:
                return x_best
            
            # Try quadratic interpolation using 3 points
            # We need 3 distinct points. Use a, b, and the best interior point
            # Or use the 3 most recent evaluations
            
            # Simple approach: use quadratic interpolation through (a,fa), (b,fb), and best point
            # If best point is a or b, use midpoint as third point
            
            if x_best == a or x_best == b:
                # Need a third point - use midpoint
                xm = (a + b) / 2.0
                fm = eval_f(xm)
                if fm is None:
                    break
                if abs(fm) <= 1e-8:
                    return xm
                # Update bracket
                if fm * fa < 0:
                    b, fb = xm, fm
                else:
                    a, fa = xm, fm
                # Update best
                if abs(fm) < abs(f_best):
                    x_best, f_best = xm, fm
                continue
            
            # Quadratic interpolation through (a, fa), (b, fb), (x_best, f_best)
            x1, f1 = a, fa
            x2, f2 = b, fb
            x3, f3 = x_best, f_best
            
            # Check for nearly equal x values
            if abs(x1 - x2) < 1e-15 * max(1.0, abs(x1)) or \
               abs(x1 - x3) < 1e-15 * max(1.0, abs(x1)) or \
               abs(x2 - x3) < 1e-15 * max(1.0, abs(x2)):
                # Fall back to bisection
                xm = (a + b) / 2.0
                fm = eval_f(xm)
                if fm is None:
                    break
                if abs(fm) <= 1e-8:
                    return xm
                if fm * fa < 0:
                    b, fb = xm, fm
                else:
                    a, fa = xm, fm
                if abs(fm) < abs(f_best):
                    x_best, f_best = xm, fm
                continue
            
            # Lagrange interpolation to find root of quadratic
            # q(x) = f1 * (x-x2)(x-x3) / ((x1-x2)(x1-x3))
            #      + f2 * (x-x1)(x-x3) / ((x2-x1)(x2-x3))
            #      + f3 * (x-x1)(x-x2) / ((x3-x1)(x3-x2))
            # Set q(x) = 0 and solve for x
            
            denom1 = (x1 - x2) * (x1 - x3)
            denom2 = (x2 - x1) * (x2 - x3)
            denom3 = (x3 - x1) * (x3 - x2)
            
            if abs(denom1) < 1e-30 or abs(denom2) < 1e-30 or abs(denom3) < 1e-30:
                xm = (a + b) / 2.0
                fm = eval_f(xm)
                if fm is None:
                    break
                if abs(fm) <= 1e-8:
                    return xm
                if fm * fa < 0:
                    b, fb = xm, fm
                else:
                    a, fa = xm, fm
                if abs(fm) < abs(f_best):
                    x_best, f_best = xm, fm
                continue
            
            # Coefficients of quadratic: Ax^2 + Bx + C = 0
            A = f1 / denom1 + f2 / denom2 + f3 / denom3
            B = -f1 * (x2 + x3) / denom1 - f2 * (x1 + x3) / denom2 - f3 * (x1 + x2) / denom3
            C = f1 * x2 * x3 / denom1 + f2 * x1 * x3 / denom2 + f3 * x1 * x2 / denom3
            
            if abs(A) < 1e-30:
                # Linear case
                if abs(B) < 1e-30:
                    xm = (a + b) / 2.0
                else:
                    x_new = -C / B
                if not math.isfinite(x_new) or not (a < x_new < b):
                    x_new = (a + b) / 2.0
            else:
                disc = B * B - 4 * A * C
                if disc < 0:
                    x_new = (a + b) / 2.0
                else:
                    sqrt_disc = math.sqrt(disc)
                    # Choose root that is in the bracket
                    x_r1 = (-B + sqrt_disc) / (2 * A)
                    x_r2 = (-B - sqrt_disc) / (2 * A)
                    if a < x_r1 < b:
                        x_new = x_r1
                    elif a < x_r2 < b:
                        x_new = x_r2
                    else:
                        x_new = (a + b) / 2.0
            
            # Safeguard: if x_new is too close to endpoint or outside bracket, use bisection
            if not (a < x_new < b):
                x_new = (a + b) / 2.0
            if abs(x_new - a) < 1e-10 * (b - a) or abs(x_new - b) < 1e-10 * (b - a):
                x_new = (a + b) / 2.0
            
            fx_new = eval_f(x_new)
            if fx_new is None:
                break
            
            if abs(fx_new) <= 1e-8:
                return x_new
            
            # Update bracket
            if fx_new * fa < 0:
                b, fb = x_new, fx_new
            else:
                a, fa = x_new, fx_new
            
            # Update best
            if abs(fx_new) < abs(f_best):
                x_best, f_best = x_new, fx_new
            
            # Check convergence
            if abs(b - a) < 1e-15 * max(1.0, abs((a + b) / 2)):
                break
        
        return x_best if abs(f_best) <= 1e-8 else None
    
    def try_secant(x1, fx1, x2, fx2, max_iter=20):
        """Secant method."""
        for i in range(max_iter):
            if time.time() > deadline:
                return None
            if abs(fx1) <= 1e-8:
                return x1
            if abs(fx2) <= 1e-8:
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
    
    # Strategy 1: Newton from x0
    result = try_newton(x, fx)
    if result is not None:
        return result
    
    # Update best
    if abs(fx) < abs(best_fx):
        best_x, best_fx = x, fx
    
    # Strategy 2: Secant with perturbed point
    h = 1e-5 * max(1.0, abs(x))
    x2 = x + h
    fx2 = eval_f(x2)
    if fx2 is not None:
        if abs(fx2) < abs(best_fx):
            best_x, best_fx = x2, fx2
        result = try_secant(x, fx, x2, fx2)
        if result is not None:
            return result
    
    # Strategy 3: Find bracket and use SIQI
    bracket = find_bracket(x, fx)
    if bracket is not None:
        a, b, fa, fb = bracket
        if abs(fa) < abs(best_fx):
            best_x, best_fx = a, fa
        if abs(fb) < abs(best_fx):
            best_x, best_fx = b, fb
        result = try_siqi(a, b, fa, fb)
        if result is not None:
            return result
    
    # Strategy 4: Multiple restarts with increasing radii
    rng = np.random.RandomState(42)
    for attempt in range(15):
        if time.time() > deadline:
            break
        if eval_count[0] >= max_evals - 10:
            break
        
        # Increasing radius for exploration
        radius = 1e-3 * (10 ** (attempt * 0.5)) * max(1.0, abs(x0))
        x_try = x0 + rng.normal(0, radius)
        fx_try = eval_f(x_try)
        if fx_try is None:
            continue
        
        if abs(fx_try) < abs(best_fx):
            best_x, best_fx = x_try, fx_try
        
        if abs(fx_try) <= 1e-8:
            return x_try
        
        # Try Newton from this point
        result = try_newton(x_try, fx_try, max_iter=10)
        if result is not None:
            return result
        
        # Try to find bracket from this point
        bracket = find_bracket(x_try, fx_try, max_expansions=10)
        if bracket is not None:
            a, b, fa, fb = bracket
            if abs(fa) < abs(best_fx):
                best_x, best_fx = a, fa
            if abs(fb) < abs(best_fx):
                best_x, best_fx = b, fb
            result = try_siqi(a, b, fa, fb)
            if result is not None:
                return result
    
    return best_x
# EVOLVE-BLOCK-END
