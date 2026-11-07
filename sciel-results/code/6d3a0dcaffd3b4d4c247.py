"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def find_root(f, x0):
    """
    Robust root finder combining Brent's method with adaptive bracketing.
    Uses a small number of function calls by leveraging a combination of
    bisection, secant, and inverse quadratic interpolation (Brent's method).
    """
    x = float(x0)
    
    # First, try to find a bracket [a, b] where f(a) and f(b) have opposite signs.
    # Start by evaluating f at x0.
    fx0 = f(x)
    if abs(fx0) <= 1e-10:
        return x
    
    # Try to find a bracket by expanding from x0
    a = x
    fa = fx0
    
    # Try several offsets to find a sign change
    offsets = [1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 4.0, -4.0, 8.0, -8.0, 0.1, -0.1, 16.0, -16.0, 0.01, -0.01]
    b = None
    fb = None
    
    for off in offsets:
        b = x0 + off
        fb = f(b)
        if abs(fb) <= 1e-10:
            return b
        if fa * fb < 0:
            break
    else:
        # No bracket found with simple offsets; try a wider search
        for scale in [32.0, 64.0, 128.0, 256.0, 512.0, 1024.0]:
            b = x0 + scale
            fb = f(b)
            if abs(fb) <= 1e-10:
                return b
            if fa * fb < 0:
                break
            b = x0 - scale
            fb = f(b)
            if abs(fb) <= 1e-10:
                return b
            if fa * fb < 0:
                break
        else:
            # Fall back to Newton-like method if no bracket found
            x = float(x0)
            for _ in range(50):
                fx = f(x)
                if abs(fx) <= 1e-10:
                    return x
                h = 1e-6 * max(1.0, abs(x))
                d = (f(x + h) - fx) / h
                if d == 0:
                    break
                x_new = x - fx / d
                # Damping if step is too large
                if abs(x_new - x) > 10 * max(1.0, abs(x)):
                    x_new = x + 0.5 * math.copysign(1.0, -fx / d) * max(1.0, abs(x))
                x = x_new
            return x
    
    # Now we have a bracket [a, b] with fa * fb < 0
    # Use Brent's method to converge
    # Ensure a < b
    if a > b:
        a, b = b, a
        fa, fb = fb, fa
    
    # Brent's method
    c = a
    fc = fa
    d = b - a
    e = d
    
    # Tolerance
    tol = 1e-12
    max_iter = 60
    
    for _ in range(max_iter):
        if abs(fb) <= 1e-10:
            return b
        if abs(fc) <= 1e-10:
            return c
        
        # Ensure b is the best estimate
        if fa * fb > 0:
            a = c
            fa = fc
            d = b - a
            e = d
        if abs(e) < tol:
            # Linear interpolation (secant)
            m = 0.5 * (a - b)
            s = fa / fb
            if abs(s) < 1e-15:
                x = b + m
            else:
                # Secant step
                x = a - m * s / (1.0 + s)
                # Clamp to bracket
                if (x - a) * (x - b) >= 0:
                    x = a + 0.5 * (b - a)
                if abs(x - b) < tol:
                    x = b + math.copysign(tol, m)
            fb = f(x)
            d = e
            e = x - b
            continue
        
        # Inverse quadratic interpolation
        s = fa / fc
        t = fb / fc
        p = fb * (a - b) * s * (s - t)
        q = fc * (b - a) * (t - 1.0) * (s - 1.0)
        # Check if p and q have the same sign
        if p * q > 0:
            # Use bisection
            x = a + 0.5 * (b - a)
        else:
            # Use interpolation
            x = b + p / q
        
        # Check if x is within bracket
        if (x - a) * (x - b) >= 0:
            x = a + 0.5 * (b - a)
        
        # Check if step is too small
        if abs(x - b) < tol:
            x = b + math.copysign(tol, a - b)
        
        fb = f(x)
        d = e
        e = x - b
    
    # Return the best estimate
    if abs(fb) < abs(fa):
        return b
    else:
        return a
# EVOLVE-BLOCK-END
