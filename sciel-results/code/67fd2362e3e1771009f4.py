"""Root finder for 3 fixed smooth polynomial equations (derivative-free IQI)."""
import math
import time

# EVOLVE-BLOCK-START
# Wall-clock safety margin: return best-so-far before the 150 s kill limit.
_DEADLINE = time.monotonic() + 110.0


def _iqi_step(a, b, c, fa, fb, fc):
    """Inverse quadratic interpolation step from 3 points. Returns None if degenerate."""
    # Guard against near-duplicate nodes.
    if (abs(a - b) < 1e-14 * max(1.0, abs(a), abs(b)) and
            abs(a - c) < 1e-14 * max(1.0, abs(a), abs(c)) and
            abs(b - c) < 1e-14 * max(1.0, abs(b), abs(c))):
        return None
    # Lagrange inverse quadratic: x = a*La + b*Lb + c*Lc where
    # La = fb*fc/((fa-fb)*(fa-fc)), etc.
    d1 = fa - fb
    d2 = fa - fc
    d3 = fb - fc
    if d1 == 0.0 or d2 == 0.0 or d3 == 0.0:
        return None
    La = (fb * fc) / (d1 * d2)
    Lb = (fa * fc) / (d1 * d3)
    Lc = (fa * fb) / (d2 * d3)
    return a * La + b * Lb + c * Lc


def _bisect_point(a, b):
    return 0.5 * (a + b)


def find_root(f, x0):
    x0 = float(x0)

    # Best-so-far tracking (x with smallest |f(x)|), used for deadline return.
    best_x = x0
    best_fx = f(x0)
    best_abs = abs(best_fx)
    if best_abs <= 1e-12:
        return best_x

    # --- Establish a bracket [lo, hi] with a sign change (f(lo)*f(hi) < 0). ---
    # Start from x0; expand outward in both directions to find opposite signs.
    lo = hi = None
    f0 = best_fx
    s0 = math.copysign(1.0, f0) if f0 != 0.0 else 0.0

    def _eval(x):
        nonlocal best_x, best_fx, best_abs
        fx = f(x)
        afx = abs(fx)
        if afx < best_abs:
            best_x, best_fx, best_abs = x, fx, afx
        return fx

    # Try to find a point with opposite sign to f0 by expanding.
    step = 1.0
    found_lo = found_hi = False
    lo_val = hi_val = None
    for _ in range(60):
        if time.monotonic() > _DEADLINE:
            return best_x
        if not found_lo:
            xl = x0 - step
            fl = _eval(xl)
            if fl == 0.0:
                return xl
            if s0 != 0.0 and math.copysign(1.0, fl) != s0:
                lo, lo_val = xl, fl
                found_lo = True
        if not found_hi:
            xh = x0 + step
            fh = _eval(xh)
            if fh == 0.0:
                return xh
            if s0 != 0.0 and math.copysign(1.0, fh) != s0:
                hi, hi_val = xh, fh
                found_hi = True
        if found_lo and found_hi:
            break
        step *= 2.0
        if step > 1e12:
            break

    # If we have a bracket, set up the IQI state.
    if found_lo and found_hi:
        # We have lo, x0, hi as three points with f(lo) and f(hi) opposite sign.
        a, b, c = lo, x0, hi
        fa, fb, fc = lo_val, f0, hi_val
        # Ensure the bracket endpoints actually bracket (they do by construction).
        lo, hi = a, c
        lo_f, hi_f = fa, fc
        # Middle point for IQI.
        mid_x, mid_f = b, fb

        # IQI + bisection loop.
        for _ in range(80):
            if time.monotonic() > _DEADLINE:
                return best_x
            if best_abs <= 1e-12:
                return best_x
            # Pick the two endpoints of the current bracket for interpolation,
            # using the point with the smaller |f| and the opposite-sign endpoint.
            # We maintain three points: lo, mid, hi. Use all three for IQI.
            xstep = _iqi_step(lo, mid_x, hi, lo_f, mid_f, hi_f)
            use_iqi = False
            if xstep is not None and math.isfinite(xstep):
                # Accept only if inside bracket and meaningfully different.
                if lo < xstep < hi and abs(xstep - mid_x) > 1e-14 * max(1.0, abs(mid_x)):
                    use_iqi = True
            if not use_iqi:
                # Bisection fallback.
                xstep = _bisect_point(lo, hi)
            fx = _eval(xstep)
            if fx == 0.0 or best_abs <= 1e-12:
                return xstep
            # Update bracket: keep the sub-interval containing a sign change.
            if math.copysign(1.0, fx) == math.copysign(1.0, lo_f):
                lo, lo_f = xstep, fx
            else:
                hi, hi_f = xstep, fx
            # Update the "middle" point to the one with smaller |f| among the two
            # interior candidates (the previous mid and the new step), to keep
            # the three-point set well-distributed for IQI.
            if abs(fx) < abs(mid_f):
                mid_x, mid_f = xstep, fx
            else:
                # keep mid; but ensure mid is inside [lo,hi]
                if not (lo < mid_x < hi):
                    mid_x, mid_f = _bisect_point(lo, hi), _eval(mid_x)
            # Convergence on interval.
            if abs(hi - lo) <= 1e-14 * max(1.0, abs(lo), abs(hi)):
                # Return the point with smallest |f|.
                cand = min((lo, hi, mid_x), key=lambda z: abs({lo: lo_f, hi: hi_f, mid_x: mid_f}[z]))
                return cand

    # --- No bracket found: fall back to a robust local search. ---
    # Use a secant-like / steepest-descent-free local minimization of |f|.
    # For a smooth polynomial, a local search from x0 usually lands on a root.
    a, b = x0 - 1.0, x0 + 1.0
    fa, fb = _eval(a), _eval(b)
    # Golden-section search on |f| over [a,b], expanding if needed.
    gr = (math.sqrt(5.0) - 1.0) / 2.0  # ~0.618
    for _ in range(200):
        if time.monotonic() > _DEADLINE:
            return best_x
        if best_abs <= 1e-12:
            return best_x
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = _eval(c)
        fd = _eval(d)
        if abs(fc) < abs(fd):
            b = d
            fb = fd
        else:
            a = c
            fa = fc
        if abs(b - a) <= 1e-13 * max(1.0, abs(a), abs(b)):
            break
    return best_x
# EVOLVE-BLOCK-END
