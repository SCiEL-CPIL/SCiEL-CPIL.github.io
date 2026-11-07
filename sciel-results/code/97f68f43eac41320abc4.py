"""Root finder for 3 fixed oscillatory equations; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from scipy.optimize import brentq, minimize_scalar

# Safety margin before the 120 s wall-clock limit.
_DEADLINE = time.time() + 110.0


def _ok(v):
    try:
        return math.isfinite(v)
    except (TypeError, ValueError):
        return False


def _illinois(f, a, b, fa, fb, max_iter=60):
    """Illinois (regula falsi with under-relaxation). Requires fa, fb finite, fa*fb<0."""
    if not _ok(fa) or not _ok(fb):
        return None
    for _ in range(max_iter):
        if time.time() > _DEADLINE:
            break
        denom = fb - fa
        if denom == 0.0:
            break
        c = (a * fb - b * fa) / denom
        if not _ok(c):
            break
        fc = f(c)
        if not _ok(fc):
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
        if abs(fb) < abs(fa):
            fa = 0.5 * fa
        else:
            fb = 0.5 * fb
        if abs(b - a) <= 1e-15 * max(1.0, abs(a), abs(b)):
            break
    if abs(fa) <= abs(fb):
        return a
    return b


def _local_secant(f, x0, max_iter=15):
    """Damped secant from x0. Returns (x, fx) if |fx|<=1e-8, else None."""
    x = float(x0)
    fx = f(x)
    if not _ok(fx):
        return None
    if abs(fx) <= 1e-8:
        return x, fx
    # Need a second point; perturb slightly
    h = max(1e-6, 1e-6 * abs(x))
    x_prev = x - h
    fx_prev = f(x_prev)
    if not _ok(fx_prev):
        x_prev = x + h
        fx_prev = f(x_prev)
        if not _ok(fx_prev):
            return None
    for _ in range(max_iter):
        if time.time() > _DEADLINE:
            return None
        denom = fx - fx_prev
        if abs(denom) < 1e-30:
            # Try a different step
            x_new = x + 0.5 * (x - x_prev)
            fx_new = f(x_new)
            if not _ok(fx_new):
                break
            x_prev, fx_prev = x, fx
            x, fx = x_new, fx_new
            continue
        x_new = x - fx * (x - x_prev) / denom
        # Damp if step is too large
        step_size = abs(x_new - x)
        if step_size > 10.0 * max(1.0, abs(x)):
            x_new = x + 0.5 * math.copysign(1.0, x_new - x) * min(step_size, 10.0 * max(1.0, abs(x)))
        if not _ok(x_new):
            break
        fx_new = f(x_new)
        if not _ok(fx_new):
            break
        if abs(fx_new) <= 1e-8:
            return x_new, fx_new
        x_prev, fx_prev = x, fx
        x, fx = x_new, fx_new
    return None


def _try_minimize(f, x0, half_width, maxiter=60):
    """Try minimize_scalar on squared residual over [x0-half_width, x0+half_width].
    Returns (x, fx) if successful and |fx| < 1e-8, else None."""
    if time.time() > _DEADLINE - 3.0:
        return None

    def obj(x):
        try:
            fx = f(x)
            if not _ok(fx):
                return 1e100
            val = fx * fx
            return val if val < 1e100 else 1e100
        except Exception:
            return 1e100

    try:
        result = minimize_scalar(
            obj,
            bounds=(x0 - half_width, x0 + half_width),
            method='bounded',
            options={'xatol': 1e-10, 'maxiter': maxiter}
        )
        if result.success and _ok(result.x):
            fx_check = f(result.x)
            if _ok(fx_check) and abs(fx_check) <= 1e-8:
                return result.x, fx_check
            elif _ok(fx_check) and abs(fx_check) < 1e-6:
                # Close enough to try a local refinement
                loc = _local_secant(f, result.x, max_iter=10)
                if loc is not None:
                    return loc
                return result.x, fx_check
    except Exception:
        pass
    return None


def find_root(f, x0):
    x0 = float(x0)
    best_x = x0
    best_fx = f(x0)
    best_abs = abs(best_fx) if _ok(best_fx) else math.inf

    if best_abs <= 1e-8:
        return best_x

    # Phase 1: Fast local secant (very few calls, high payoff)
    loc = _local_secant(f, x0)
    if loc is not None:
        return loc[0]

    # Phase 2: Try minimize_scalar on a tight local window first
    # This is robust for oscillatory functions and avoids expensive bracketing
    base_scale = max(1.0, abs(x0))
    for hw_factor in [0.5, 1.0, 2.0]:
        hw = base_scale * hw_factor
        if hw < 1e-4:
            hw = 1e-4
        res = _try_minimize(f, x0, hw, maxiter=50)
        if res is not None:
            if res[1] is not None and abs(res[1]) <= 1e-8:
                return res[0]
            # Update best if improved
            if res[1] is not None and abs(res[1]) < best_abs:
                best_x, best_fx, best_abs = res[0], res[1], abs(res[1])

    # Phase 3: Bracket expansion to find sign change
    step = base_scale
    found_sign = False
    ca = None
    cb = None
    fa = best_fx
    for i in range(20):
        if time.time() > _DEADLINE:
            break
        step *= 2.0
        if step > 1e12:
            break
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

    if found_sign:
        try:
            root = brentq(f, ca, cb, xtol=1e-12, rtol=1e-14, maxiter=80)
            if _ok(root):
                return root
        except Exception:
            pass
        fa2 = f(ca)
        fb2 = f(cb)
        r = _illinois(f, ca, cb, fa2, fb2)
        if r is not None and _ok(r):
            return r

    # Phase 4: Refine around best point with local search
    if _ok(best_fx):
        h = max(1e-4, 1e-4 * abs(best_x))
        for k in range(1, 10):
            if time.time() > _DEADLINE:
                break
            off = h * (2.0 ** k)
            for sgn in (1.0, -1.0):
                p = best_x + sgn * off
                fp = f(p)
                if not _ok(fp):
                    continue
                if abs(fp) < best_abs:
                    best_abs, best_x, best_fx = abs(fp), p, fp
                if best_fx * fp < 0.0:
                    lo, hi = (best_x - sgn * off, best_x + sgn * off) if sgn > 0 else (best_x - sgn * off, best_x + sgn * off)
                    lo, hi = min(lo, hi), max(lo, hi)
                    try:
                        root = brentq(f, lo, hi, xtol=1e-12, rtol=1e-14, maxiter=60)
                        if _ok(root):
                            return root
                    except Exception:
                        pass
                    r = _illinois(f, lo, hi, f(lo), f(hi))
                    if r is not None and _ok(r):
                        return r
                    break

    # Phase 5: Wider minimize_scalar as final fallback
    if time.time() < _DEADLINE - 5.0:
        for hw_factor in [5.0, 20.0, 100.0]:
            if time.time() > _DEADLINE - 5.0:
                break
            hw = base_scale * hw_factor
            if hw < 1e-3:
                hw = 1e-3
            if hw > 1e6:
                hw = 1e6
            res = _try_minimize(f, best_x, hw, maxiter=40)
            if res is not None:
                if abs(res[1]) <= 1e-8:
                    return res[0]
                if abs(res[1]) < best_abs:
                    best_x, best_fx, best_abs = res[0], res[1], abs(res[1])

    if _ok(best_x):
        return best_x
    return x0
# EVOLVE-BLOCK-END
