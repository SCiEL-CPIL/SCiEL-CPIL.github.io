import math
import time

_START_TIME = time.time()
_WALL_LIMIT = 115.0  # safety margin before 120s

def _time_left():
    return _WALL_LIMIT - (time.time() - _START_TIME)

def _is_finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)

def find_root(f, x0):
    x = float(x0)
    
    # Check initial point
    try:
        fx = f(x)
    except Exception:
        return x
    if not _is_finite(fx):
        return x
    if abs(fx) <= 1e-8:
        return x
    
    best_x = x
    best_fx = fx
    
    # --- Phase 1: Global Bracketing Search ---
    # Expand a window around x0 to find a sign change.
    # This handles cases where Newton/secant might diverge or get stuck.
    a = x
    fa = fx
    b = None
    fb = None
    
    step = max(1e-4, abs(x) * 1e-4)
    # Try expanding in both directions to find a bracket [a, b] with f(a)*f(b) < 0
    # We limit the expansion to avoid excessive calls if no root is nearby,
    # but for smooth polynomials with a root, this should find it quickly.
    max_expansions = 20
    
    found_bracket = False
    for i in range(max_expansions):
        if _time_left() < 2.0:
            break
            
        # Try positive direction
        b_candidate = a + step * (i + 1)
        try:
            fb_candidate = f(b_candidate)
        except Exception:
            break
        if _is_finite(fb_candidate):
            if fa * fb_candidate < 0:
                b = b_candidate
                fb = fb_candidate
                found_bracket = True
                break
            # Keep the best point if no sign change yet
            if abs(fb_candidate) < abs(best_fx):
                best_x = b_candidate
                best_fx = fb_candidate
        
        # Try negative direction
        b_candidate2 = a - step * (i + 1)
        try:
            fb_candidate2 = f(b_candidate2)
        except Exception:
            break
        if _is_finite(fb_candidate2):
            if fa * fb_candidate2 < 0:
                b = b_candidate2
                fb = fb_candidate2
                found_bracket = True
                break
            if abs(fb_candidate2) < abs(best_fx):
                best_x = b_candidate2
                best_fx = fb_candidate2

    if not found_bracket:
        # If no bracket found, fall back to secant/Newton-like local search
        # or just return best_x. Given the task guarantees roots, this is a fallback.
        # Let's try a simple secant step from best_x and x0 if they are different.
        if best_x != x:
            x1 = x
            x2 = best_x
            f1 = fx
            f2 = best_fx
            for i in range(50):
                if _time_left() < 2.0:
                    break
                if f1 == f2:
                    break
                x_new = x2 - f2 * (x2 - x1) / (f2 - f1)
                if not _is_finite(x_new):
                    break
                try:
                    f_new = f(x_new)
                except Exception:
                    break
                if not _is_finite(f_new):
                    break
                if abs(f_new) < abs(best_fx):
                    best_x = x_new
                    best_fx = f_new
                if abs(f_new) <= 1e-8:
                    return x_new
                x1, f1 = x2, f2
                x2, f2 = x_new, f_new
        return best_x

    # --- Phase 2: Inverse Cubic Interpolation ---
    # We have a bracket [a, b] with fa*fb < 0.
    # We need 3 or 4 points to do cubic interpolation.
    # Currently we have a, b. Let's get a third point, say the midpoint or a golden section point.
    # To minimize calls, let's use the midpoint first to get a third point.
    m = 0.5 * (a + b)
    try:
        fm = f(m)
    except Exception:
        return best_x
    if not _is_finite(fm):
        return best_x
    if abs(fm) <= 1e-8:
        return m
    
    # Update bracket to the smaller sub-interval
    if fa * fm < 0:
        b = m
        fb = fm
    else:
        a = m
        fa = fm
        
    # Now we have 3 points: a, m (old midpoint, now inside bracket), b
    # Wait, the points are not necessarily sorted or distinct enough for a good cubic.
    # Let's collect points.
    # Points: (a, fa), (m, fm), (b, fb) where fa*fb < 0.
    # Actually, m is either a or b now? No, m was the midpoint.
    # If fa*fm < 0, then b becomes m. So points are a, m, b(old).
    # Let's just maintain a list of points and use the last 3 for cubic.
    
    # Let's restart the interpolation loop with the known bracket [a, b] and fa, fb.
    # We need one more point to start cubic interpolation properly, or use quadratic.
    # Let's use the midpoint again to get a 4th point if needed, or just use quadratic first.
    # The "Knowledge" says inverse cubic is 1 call/iteration.
    # Standard approach:
    # Start with 2 points. Add a 3rd (midpoint).
    # Then iterate:
    #   Use last 3 points to fit cubic (or quadratic if 3 points).
    #   Find root of interpolant.
    #   Evaluate f at that root.
    #   Add new point.
    #   Shrink bracket.
    
    # Let's refine the points.
    # Current known: a, b with fa*fb < 0.
    # We already evaluated m.
    # If m is inside (a,b), we have 3 points.
    # Let's ensure a < b.
    if a > b:
        a, b = b, a
        fa, fb = fb, fa
        # m is still the same value, but its position relative to a,b matters.
        # We don't need to reorder m's value, just its x.
    
    # Points: (a, fa), (b, fb). We have m and fm.
    # Let's add m to our set of points.
    # For the first step, let's just do a secant step to get a new point, then switch to cubic.
    # Or better, use the 3 points (a, m, b) to do a quadratic interpolation root.
    
    # Let's implement a robust loop:
    # Maintain a list of recent points.
    # Start with points: (a, fa), (b, fb).
    # Get a third point: let's try the midpoint again? No, we have m.
    # Let's just use the 3 points (a, m, b) to find a root via quadratic interpolation.
    
    # Quadratic interpolation root:
    # Given (x0, f0), (x1, f1), (x2, f2), the root of the parabola is:
    # x = x0 + 0.5 * ( (f0/(f0-f1))*(x0-x1) + (f0/(f0-f2))*(x0-x2) )  -- This is not quite right.
    # Let's use the standard formula for the root of a quadratic passing through 3 points.
    # Or simpler: use Newton's method with finite difference if we don't want to mess with cubic formulas.
    # But the prompt highlights cubic interpolation.
    
    # Let's use the formula for the root of the interpolating polynomial.
    # For 3 points, it's a quadratic.
    # Let's just use the secant method for the first few steps to get close, then switch?
    # No, let's stick to the "Knowledge": Inverse cubic interpolation.
    # It requires 4 points. We have 3 (a, m, b). Let's get a 4th by doing a secant step from the best 2.
    
    # Actually, a simpler and very effective method is the **Illinois algorithm** or **Brent's method**.
    # Brent's method combines bisection, secant, and inverse quadratic interpolation.
    # It is guaranteed to converge and is very efficient (typically ~1 call/iteration).
    # Let's implement a simplified Brent's method or just use Secant + Bisection.
    
    # Given the "Knowledge" specifically mentions "Inverse cubic interpolation", let's try to implement that.
    # But implementing a robust cubic interpolator from scratch is error-prone.
    # SciPy is available. `scipy.optimize.brentq` is the gold standard for bracketed root finding.
    # It uses Brent's method.
    # Does `brentq` count calls? Yes, it calls `f`.
    # Is it efficient? Yes, typically 10-20 calls for high precision.
    # The current program uses ~50-100 calls.
    # Using `scipy.optimize.brentq` is a MATERIALLY DIFFERENT strategy (using a robust library function for bracketed root finding) compared to the hand-rolled Newton/Secant.
    
    # Let's use scipy.optimize.brentq.
    # We have a bracket [a, b].
    
    try:
        from scipy.optimize import brentq
    except ImportError:
        # Fallback to secant if scipy is not available (though it is stated to be available)
        return best_x
    
    # Ensure a < b
    if a > b:
        a, b = b, a
        fa, fb = fb, fa
        
    # Check if a or b is a root
    if abs(fa) <= 1e-8:
        return a
    if abs(fb) <= 1e-8:
        return b
    
    # Use brentq
    # xtol=1e-12, rtol=1e-12 to ensure |f(x)| is small.
    # Actually, brentq stops when the interval is small, not when f(x) is small.
    # For smooth polynomials, a small interval implies a small f(x) if the derivative is not zero.
    # We should check the result.
    
    try:
        root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
    except Exception:
        return best_x
        
    if _is_finite(root):
        try:
            f_root = f(root)
            if _is_finite(f_root) and abs(f_root) <= 1e-8:
                return root
        except Exception:
            pass
    
    return best_x
