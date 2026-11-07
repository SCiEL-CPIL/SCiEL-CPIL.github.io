"""Root finder for 3 fixed oscillatory equations; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from scipy.optimize import brentq

# Safety margin before the 120 s wall-clock limit.
_DEADLINE = time.time() + 110.0


def _ok(v):
    try:
        return math.isfinite(v)
    except (TypeError, ValueError):
        return False


def _illinois(f, a, b, fa, fb, max_iter=80):
    """Illinois (regula falsi with under-relaxation). Requires fa, fb finite, fa*fb<0."""
    if not _ok(fa) or not _ok(fb):
        return None
    for _ in range(max_iter):
        if time.time() > _DEADLINE:
            break
        # secant estimate
        denom = fb - fa
        if denom == 0.0:
            break
        c = (a * fb - b * fa) / denom
        if not _ok(c):
            break
        fc = f(c)
        if not _ok(fc):
            # nudge c toward the interior
            c = 0.5 * (a + b)
            fc = f(c)
            if not _ok(fc):
                break
        if abs(fc) <= 1e-12:
            return c
        if fa * fc < 0.0:
            b, fb = c, fc
        else:
            a, fa = c, fc
        # Illinois damping: keep the bracket, shrink the side that changed less
        if abs(fb) < abs(fa):
            fa = 0.5 * fa
        else:
            fb = 0.5 * fb
        if abs(b - a) <= 1e-15 * max(1.0, abs(a), abs(b)):
            break
    # return the endpoint with the smaller |f|
    if abs(fa) <= abs(fb):
        return a
    return b


def _local(f, x0):
    """Fast local Illinois/secant from x0. Returns (x, fx) or None."""
    x = float(x0)
    fx = f(x)
    if not _ok(fx):
        return None
    if abs(fx) <= 1e-8:
        return x, fx
    x_prev = x
    fx_prev = fx
    for _ in range(12):
        if time.time() > _DEADLINE:
            return None
        # secant step
        denom = fx - fx_prev
        if denom == 0.0 or not _ok(denom):
            break
        x_new = x - fx * (x - x_prev) / denom
        if not _ok(x_new):
            break
        fx_new = f(x_new)
        if not _ok(fx_new):
            break
        if abs(fx_new) <= 1e-8:
            return x_new, fx_new
        # divergence guard
        if abs(x_new - x) > 1000.0 * (1.0 + abs(x)):
            break
        x_prev, fx_prev = x, fx
        x, fx = x_new, fx_new
    return None


def find_root(f, x0):
    x0 = float(x0)
    best_x = x0
    best_fx = f(x0)
    best_abs = abs(best_fx) if _ok(best_fx) else math.inf

    # Phase 1: fast local method (few calls, high payoff when it works).
    loc = _local(f, x0)
    if loc is not None:
        return loc[0]

    # Phase 2: build a bracket around x0.
    # Try expanding a bracket to find a sign change or a local |f| minimum.
    scale = max(1.0, abs(x0))
    # scan outward with a modest geometric factor; keep the best |f| point.
    a = x0
    fa = best_fx
    b = x0
    fb = best_fx

    # Expand right and left to find a sign change (guaranteed root) or local min.
    step = scale
    found_sign = False
    ca = None
    cb = None
    for _ in range(30):
        if time.time() > _DEADLINE:
            break
        step *= 2.5
        r = x0 + step
        fr = f(r)
        if _ok(fr):
            if abs(fr) < best_abs:
                best_abs, best_x, best_fx = abs(fr), r, fr
            if _ok(fa) and fa * fr < 0.0:
                ca, cb = x0, r
                found_sign = True
                break
        l = x0 - step
        fl = f(l)
        if _ok(fl):
            if abs(fl) < best_abs:
                best_abs, best_x, best_fx = abs(fl), l, fl
            if _ok(fa) and fa * fl < 0.0:
                ca, cb = l, x0
                found_sign = True
                break
        if step > 1e12:
            break

    if found_sign:
        try:
            root = brentq(f, ca, cb, xtol=1e-12, rtol=1e-14, maxiter=100)
            if _ok(root):
                return root
        except Exception:
            pass
        # fallback to Illinois on the same bracket
        fa2 = f(ca)
        fb2 = f(cb)
        r = _illinois(f, ca, cb, fa2, fb2)
        if r is not None and _ok(r):
            return r

    # No sign change found: try to refine around the best |f| point (touching root).
    # Build a small bracket around best_x and look for a local minimum / sign change.
    cx = best_x
    cfx = best_fx
    if _ok(cfx):
        h = max(1e-4, 1e-4 * abs(cx))
        # sample a few offsets to find a sign change or a lower point
        for k in range(1, 12):
            if time.time() > _DEADLINE:
                break
            off = h * (2.0 ** k)
            for sgn in (1.0, -1.0):
                p = cx + sgn * off
                fp = f(p)
                if not _ok(fp):
                    continue
                if abs(fp) < best_abs:
                    best_abs, best_x, best_fx = abs(fp), p, fp
                if _ok(cfx) and cfx * fp < 0.0:
                    lo, hi = (cx, p) if sgn > 0 else (p, cx)
                    try:
                        root = brentq(f, lo, hi, xtol=1e-12, rtol=1e-14, maxiter=100)
                        if _ok(root):
                            return root
                    except Exception:
                        pass
                    r = _illinois(f, lo, hi, f(lo), f(hi))
                    if r is not None and _ok(r):
                        return r
                    break
    # Return the best finite candidate found.
    if _ok(best_x):
        return best_x
    return x0
# EVOLVE-BLOCK-END
