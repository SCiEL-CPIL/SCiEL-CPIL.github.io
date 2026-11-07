"""Safeguarded Secant Method root finder."""
# EVOLVE-BLOCK-START
import math

def find_root(f, x0):
    x0 = float(x0)
    if not math.isfinite(x0):
        x0 = 0.0

    fa = f(x0)
    if abs(fa) <= 1e-8:
        return x0
    if not math.isfinite(fa):
        x0 = 0.0
        fa = f(x0)
        if abs(fa) <= 1e-8:
            return x0

    a = x0
    b = x0 + 1.0
    fb = f(b)
    
    if abs(fb) <= 1e-8:
        return b
    if not math.isfinite(fb):
        b = x0 - 1.0
        fb = f(b)
        if abs(fb) <= 1e-8:
            return b

    # Bracketing
    if fa * fb > 0:
        # Expand bracket
        # Try expanding in both directions
        for _ in range(50):
            # Expand the side with larger |f|? Or just alternate?
            # Let's expand the side that is further from x0
            if abs(b - x0) >= abs(a - x0):
                b = b + (b - a)
                fb = f(b)
                if abs(fb) <= 1e-8:
                    return b
                if not math.isfinite(fb):
                    break
            else:
                a = a - (b - a)
                fa = f(a)
                if abs(fa) <= 1e-8:
                    return a
                if not math.isfinite(fa):
                    break
            
            if math.isfinite(fa) and math.isfinite(fb) and fa * fb < 0:
                break
        
        if not (math.isfinite(fa) and math.isfinite(fb) and fa * fb < 0):
            # Fallback: return best guess
            return a if abs(fa) <= abs(fb) else b

    # Safeguarded Secant Method
    # Invariant: f(a) * f(b) < 0
    for _ in range(100):
        if abs(fa) <= 1e-8:
            return a
        if abs(fb) <= 1e-8:
            return b

        # Ensure a is the "better" endpoint (smaller |f|)
        if abs(fa) > abs(fb):
            a, b = b, a
            fa, fb = fb, fa

        # Calculate secant step
        if fa != fb:
            x_new = b - fb * (b - a) / (fb - fa)
        else:
            x_new = 0.5 * (a + b)

        # Safeguard: if x_new is not in (a, b), use bisection
        if not (a < x_new < b):
            x_new = 0.5 * (a + b)
        
        # Additional safeguard: if the step is too small, bisect
        # This helps with convergence near the root
        if abs(x_new - b) < 1e-12 * abs(b - a):
            x_new = 0.5 * (a + b)

        f_new = f(x_new)
        
        if abs(f_new) <= 1e-8:
            return x_new
        if not math.isfinite(f_new):
            # Fallback to bisection
            x_new = 0.5 * (a + b)
            f_new = f(x_new)
            if not math.isfinite(f_new):
                return a if abs(fa) <= abs(fb) else b

        # Update bracket
        if fa * f_new < 0:
            b = x_new
            fb = f_new
        else:
            a = x_new
            fa = f_new

    return a if abs(fa) <= abs(fb) else b
# EVOLVE-BLOCK-END
