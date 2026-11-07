"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """Safeguarded inverse interpolation (Regula Falsi with Brent's safeguard).
    Uses 1 function call per iteration after initial bracketing, vs 2 for Newton."""
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s kill
    
    x = float(x0)
    fx = f(x)
    if abs(fx) <= 1e-8:
        return x
    
    # Phase 1: Find a bracket [a, b] with f(a)*f(b) <= 0
    a, b = x, x
    fa, fb = fx, fx
    step = max(1e-4, abs(x) * 1e-4)
    
    # Try expanding bracket geometrically
    for i in range(60):
        if time.time() > deadline:
            break
        # Try x + step
        x_right = x + step
        fx_right = f(x_right)
        if abs(fx_right) <= 1e-8:
            return x_right
        if fx * fx_right < 0:
            a, b = x, x_right
            fa, fb = fx, fx_right
            break
        # Try x - step
        x_left = x - step
        fx_left = f(x_left)
        if abs(fx_left) <= 1e-8:
            return x_left
        if fx * fx_left < 0:
            a, b = x_left, x
            fa, fb = fx_left, fx
            break
        step *= 2.0
    else:
        # No bracket found; fall back to Newton
        return _newton_fallback(f, x, fx, deadline)
    
    # Phase 2: Safeguarded inverse interpolation
    # Keep track of best (smallest |f|)
    best_x = x if abs(fx) < abs(fa) else a
    best_fx = min(abs(fx), abs(fa))
    
    # Ensure |fa| <= |fb| so we always interpolate from the endpoint with smaller |f|
    if abs(fa) > abs(fb):
        a, b = b, a
        fa, fb = fb, fa
        best_x = a if abs(fa) < abs(fa) else b  # recompute below
    
    # Track the two most recent points for inverse interpolation
    # We maintain a bracket [a, b] and a previous point c
    # Use Brent's strategy: try inverse quadratic/linear interpolation, fall back to bisection
    
    # Initialize: we have a, b with fa*fb <= 0
    # c is the previous iterate (initially None)
    c = None
    fc = None
    prev_a, prev_b = a, b
    prev_fa, prev_fb = fa, fb
    
    max_iter = 100
    for i in range(max_iter):
        if time.time() > deadline:
            break
        if abs(fa) < abs(fb):
            # Swap so that |fa| <= |fb|
            a, b = b, a
            fa, fb = fb, fa
        
        # Check convergence
        if abs(fa) <= 1e-8:
            return a
        if abs(fb) <= 1e-8:
            return b
        
        # Update best
        if abs(fa) < best_fx:
            best_x = a
            best_fx = abs(fa)
        if abs(fb) < best_fx:
            best_x = b
            best_fx = abs(fb)
        
        # Try inverse interpolation
        s = None
        use_interp = False
        
        if c is not None and abs(c - a) > 1e-15 and abs(c - b) > 1e-15:
            # Inverse quadratic interpolation (Brent's)
            # Using the three points a, b, c
            # Only use if c was the previous value of a or b
            if abs(fc) < abs(fa):
                # c is closer to root than a
                # Try inverse quadratic
                try:
                    s = _inverse_quadratic(a, fa, b, fb, c, fc)
                except (ZeroDivisionError, OverflowError):
                    s = None
                if s is not None:
                    use_interp = True
            else:
                # Inverse linear interpolation (Regula Falsi)
                if abs(fa - fb) > 1e-300:
                    s = b - fb * (b - a) / (fb - fa)
                    use_interp = True
        else:
            # Only have two points: use Regula Falsi
            if abs(fa - fb) > 1e-300:
                s = b - fb * (b - a) / (fb - fa)
                use_interp = True
        
        # Safeguard: check if s is in the bracket and not too close to endpoints
        if use_interp and s is not None:
            # s should be between a and b
            if (s - a) * (s - b) < 0:  # strictly inside
                # Check it's not too close to a or b
                if abs(s - a) > 1e-15 * max(1.0, abs(a)) and abs(s - b) > 1e-15 * max(1.0, abs(b)):
                    pass  # use s
                else:
                    use_interp = False
            else:
                use_interp = False
        
        if not use_interp or s is None:
            # Bisection fallback
            s = (a + b) / 2.0
            if abs(s - a) < 1e-15 or abs(s - b) < 1e-15:
                # Bracket is too small
                break
        
        fs = f(s)
        if abs(fs) <= 1e-8:
            return s
        
        # Update bracket
        if fs * fa < 0:
            # Root is in [a, s]
            b, fb = s, fs
        else:
            # Root is in [s, b]
            a, fa = s, fs
        
        # Update c for next iteration (c is the previous iterate that is no longer in bracket)
        # In Brent's method, c is set to the point that was replaced
        if fs * fa < 0:
            # b was replaced, so c = old b
            if c is None or abs(c - b) < abs(c - a):
                c, fc = b, fb
            else:
                c, fc = b, fb
        else:
            # a was replaced, so c = old a
            c, fc = a, fa
        
        # If bracket is shrinking too slowly, force bisection next
        if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
            break
    
    # Return best found
    if abs(fa) <= 1e-8:
        return a
    if abs(fb) <= 1e-8:
        return b
    return best_x


def _inverse_quadratic(x0, f0, x1, f1, x2, f2):
    """Inverse quadratic interpolation through three points.
    Returns the x-value where the interpolating polynomial crosses zero."""
    # Lagrange form for inverse interpolation: x = f * L(f)
    # where L is the Lagrange basis in terms of f
    d0 = f1 - f2
    d1 = f0 - f2
    d2 = f0 - f1
    if abs(d0) < 1e-300 or abs(d1) < 1e-300 or abs(d2) < 1e-300:
        return None
    # x(f) = f0 * (f - f1)(f - f2) / ((f0-f1)(f0-f2)) + f1 * (f - f0)(f - f2) / ((f1-f0)(f1-f2)) + f2 * (f - f0)(f - f1) / ((f2-f0)(f2-f1))
    # At f=0:
    # x(0) = f0 * f1*f2 / ((f0-f1)(f0-f2)) + f1 * f0*f2 / ((f1-f0)(f1-f2)) + f2 * f0*f1 / ((f2-f0)(f2-f1))
    term0 = f0 * f1 * f2 / ((f0 - f1) * (f0 - f2))
    term1 = f1 * f0 * f2 / ((f1 - f0) * (f1 - f2))
    term2 = f2 * f0 * f1 / ((f2 - f0) * (f2 - f1))
    s = term0 + term1 + term2
    if not math.isfinite(s):
        return None
    return s


def _newton_fallback(f, x, fx, deadline):
    """Newton's method as fallback if bracketing fails."""
    for _ in range(40):
        if time.time() > deadline:
            break
        if abs(fx) <= 1e-8:
            return x
        h = 1e-6 * max(1.0, abs(x))
        d = (f(x + h) - fx) / h
        if d == 0:
            break
        x_new = x - fx / d
        if not math.isfinite(x_new):
            break
        fx = f(x_new)
        x = x_new
    return x
# EVOLVE-BLOCK-END
