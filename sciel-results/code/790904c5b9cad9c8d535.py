"""Derivative-free safeguarded inverse-interpolation root finder."""
# EVOLVE-BLOCK-START
import math

def find_root(f, x0):
    # Work entirely in floats; guard against non-finite inputs.
    x0 = float(x0)
    if not math.isfinite(x0):
        x0 = 0.0

    # --- Step 1: cheap bracketing search (max 60 evaluations) ---
    # Reuse the starting evaluation so we don't waste a call.
    fa = f(x0)
    if abs(fa) <= 1e-8:
        return x0
    if not math.isfinite(fa):
        x0 = 0.0
        fa = f(x0)
        if abs(fa) <= 1e-8:
            return x0

    a = x0
    # Expand a bracket [a, b] until f changes sign.
    b = x0 + 1.0
    fb = f(b)
    if abs(fb) <= 1e-8:
        return b
    if not math.isfinite(fb):
        b = x0 - 1.0
        fb = f(b)
        if abs(fb) <= 1e-8:
            return b

    for _ in range(58):
        if math.isfinite(fa) and math.isfinite(fb) and fa * fb < 0.0:
            break
        # Expand the side whose endpoint is farther from the other.
        if abs(b - x0) >= abs(a - x0):
            b = b + (b - a)
            fb = f(b)
        else:
            a = a - (b - a)
            fa = f(a)
        if abs(fa) <= 1e-8:
            return a
        if abs(fb) <= 1e-8:
            return b
        if not math.isfinite(fa):
            a = x0
            fa = f(a)
        if not math.isfinite(fb):
            b = x0 + 1.0
            fb = f(b)
    else:
        # No sign change found; return the better of the two endpoints.
        return a if abs(fa) <= abs(fb) else b

    # --- Step 2: safeguarded inverse interpolation ---
    # Invariant: f(a) * f(b) <= 0, and we keep the two most recent
    # evaluations (a, fa) and (b, fb).
    for _ in range(60):
        if abs(fa) <= 1e-8:
            return a
        if abs(fb) <= 1e-8:
            return b
        # Ensure a is the "old" endpoint (smaller |f|) and b the "new".
        if abs(fa) > abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        # a is now the better (smaller |f|) endpoint; b the worse.
        m = 0.5 * (a + b)
        fm = f(m)
        if abs(fm) <= 1e-8:
            return m
        if not math.isfinite(fm):
            # Midpoint eval failed; fall back to pure bisection.
            a, b = b, m
            fa, fb = fb, fm
            continue
        # Update bracket.
        if fa * fm < 0.0:
            b, fb = m, fm
        else:
            a, fa = m, fm
        # Inverse linear interpolation (Regula Falsi) candidate.
        if fa != fb:
            xi = b - fb * (b - a) / (fb - fa)
        else:
            xi = m
        # Safeguard: accept xi only if it is a genuine interpolation
        # step that stays strictly inside the bracket; else bisect.
        if not (a < xi < b) or (abs(xi - b) >= 0.5 * abs(b - a)):
            xi = m
        # Evaluate the chosen point.
        fxi = f(xi)
        if abs(fxi) <= 1e-8:
            return xi
        if not math.isfinite(fxi):
            a, b = b, m
            fa, fb = fb, fm
            continue
        if fa * fxi < 0.0:
            b, fb = xi, fxi
        else:
            a, fa = xi, fxi
    # Return the endpoint with the smaller residual.
    return a if abs(fa) <= abs(fb) else b
# EVOLVE-BLOCK-END
