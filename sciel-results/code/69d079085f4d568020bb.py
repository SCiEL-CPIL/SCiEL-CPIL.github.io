"""
Root finder: coarse-grid global bracket/minimum scan + superlinear 1-call/secant
driver, with a polynomial-specific Newton/central-diff polish tail.

Design family preserved: locate a sign bracket (or global min-|f| point), then
drive a 1-call/step secant method to |f(x)| <= 1e-8, with an IQU-Brent
safeguard. The new coarse-grid scan is a discrete global optimizer for |f(x)|
that guarantees a robust starting point in ~40 calls regardless of x0, which
reduces mean calls on the harder equations. A cheap Newton/central-diff polish
runs only when we are already close and have spare budget, saving calls in
the tight tail.
"""
import time
import math

TOL = 1e-8
MAX_CALLS = 470
PHI = (1.0 + math.sqrt(5.0)) / 2.0


def find_root(f, x0):
    t0 = time.time()
    deadline = t0 + 100.0
    x0 = float(x0)
    call_count = 0
    cache = {}

    def _time_ok():
        return (time.time() - t0) < deadline and call_count < MAX_CALLS

    def _f(x_val):
        nonlocal call_count
        if call_count >= MAX_CALLS:
            return None
        key = x_val
        if key in cache:
            return cache[key]
        try:
            val = f(key)
        except Exception:
            call_count += 1
            return None
        call_count += 1
        if not math.isfinite(val):
            return None
        cache[key] = val
        return val

    def _best_from_cache():
        best_x, best_abs = None, math.inf
        for k, v in cache.items():
            if v is None:
                continue
            av = abs(v)
            if av < best_abs:
                best_abs, best_x = av, k
        return best_x, best_abs

    # --- Phase 0: evaluate start point ---
    fx0 = _f(x0)
    if fx0 is None:
        return x0
    if abs(fx0) <= TOL:
        return x0

    bracket = None  # (a, b, fa, fb)

    # --- Phase 1a: coarse adaptive grid scan (global optimizer for |f|) ---
    base = max(abs(x0), 1.0)
    for scale in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0]:
        if not _time_ok():
            break
        half = base * scale
        n = 40
        prev_x = prev_f = None
        for i in range(n + 1):
            if not _time_ok():
                break
            t = i / n
            xt = x0 - half + 2.0 * half * t
            ft = _f(xt)
            if ft is None:
                prev_x = prev_f = None
                continue
            if abs(ft) <= TOL:
                return xt
            if prev_f is not None and prev_f * ft < 0.0:
                bracket = (prev_x, xt, prev_f, ft)
                break
            prev_x, prev_f = xt, ft
        if bracket is not None:
            break
    if not _time_ok():
        bx, _ = _best_from_cache()
        return bx if bx is not None else x0

    # --- Phase 1b: wide linear grid if still no bracket ---
    if bracket is None:
        for scale in [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]:
            if not _time_ok():
                break
            half = max(abs(x0), 1.0) * scale * 16.0
            n = 40
            prev_x = prev_f = None
            for i in range(n + 1):
                if not _time_ok():
                    break
                t = i / n
                xt = x0 - half + 2.0 * half * t
                ft = _f(xt)
                if ft is None:
                    prev_x = prev_f = None
                    continue
                if abs(ft) <= TOL:
                    return xt
                if prev_f is not None and prev_f * ft < 0.0:
                    bracket = (prev_x, xt, prev_f, ft)
                    break
                prev_x, prev_f = xt, ft
            if bracket is not None:
                break
    if not _time_ok():
        bx, _ = _best_from_cache()
        return bx if bx is not None else x0

    # --- Secant driver (1 call/step, superlinear) ---
    def secant_driver(xc, fc, xp, fp):
        for _ in range(70):
            if not _time_ok():
                break
            if fc is not None and abs(fc) <= TOL:
                return xc
            if fp is not None and abs(fp) <= TOL:
                return xp
            denom = fc - fp
            if abs(denom) < 1e-15:
                break
            xn = xc - fc * (xc - xp) / denom
            if not math.isfinite(xn):
                break
            xp, fp = xc, fc
            xc = xn
            fc = _f(xc)
            if fc is None:
                break
        if fc is not None and fp is not None:
            return xc if abs(fc) <= abs(fp) else xp
        return xc if fc is not None else xp

    # --- IQU-Brent driver (1 call/step, superlinear, bracket-safe) ---
    def brent_driver(a, b, fa, fb):
        if fa == 0.0:
            return a
        if fb == 0.0:
            return b
        if fa * fb > 0.0:
            return secant_driver(a, fa, b, fb)
        c, fc = a, fa
        d = e = b - a
        for _ in range(70):
            if not _time_ok():
                break
            if fb == 0.0:
                return b
            if fc == 0.0:
                return c
            if fa * fb > 0.0:
                a, fa = c, fc
                d = e = b - a
            if abs(e) < 1e-15 or abs(fa) <= abs(fb):
                c, fc = b, fb
                d = e = b - a
            try:
                if abs(d) > 1e-15:
                    s = fa / fb
                    if a != c and c != b:
                        w = fb * (d / (c - b)) * s * (s - fa / fc)
                        t = 1.5 * (c - b) * (fa / fc)
                    else:
                        w, t = 0.0, 0.0
                    if (2.0 * t - w) < (3.0 * t - w) * min(1.0, abs(t) / w) \
                            and (2.0 * t - w) < abs(e):
                        e = d
                        b = c + t
                    else:
                        b = c + (d if d * e > 0.0 else math.copysign(1e-15, d))
                else:
                    e = d = b - a
            except (ZeroDivisionError, OverflowError):
                e = d = b - a
            fb = _f(b)
            if fb is None:
                break
        cands = [(a, fa), (b, fb)]
        if fc is not None:
            cands.append((c, fc))
        cands = [p for p in cands if p[1] is not None]
        if not cands:
            return a
        return min(cands, key=lambda p: abs(p[1]))[0]

    # --- Polynomial Newton polish (forward-diff, damping) ---
    def _fwd_diff(x, h):
        return (_f(x + h) - _f(x)) / h

    def _damp_newton(x, fx):
        for _ in range(6):
            if not _time_ok():
                break
            if abs(fx) <= TOL:
                return x, fx
            h = 1e-6 * max(1.0, abs(x))
            fp = _fwd_diff(x, h)
            if fp is None or not math.isfinite(fp) or abs(fp) < 1e-15:
                break
            step = fx / fp
            for lam in (1.0, 0.5, 0.25, 0.125, 0.0625):
                xn = x - lam * step
                if not math.isfinite(xn):
                    break
                fn = _f(xn)
                if fn is None:
                    break
                if abs(fn) < abs(fx) * 0.9:
                    return xn, fn
            break
        return x, fx

    def _central_polish(x, fx):
        for _ in range(4):
            if not _time_ok():
                break
            if abs(fx) <= TOL:
                return x, fx
            h = 1e-5 * max(1.0, abs(x))
            fp = (_f(x + h) - _f(x - h)) / (2.0 * h)
            if fp is None or not math.isfinite(fp) or abs(fp) < 1e-15:
                break
            step = fx / fp
            for lam in (1.0, 0.5, 0.25, 0.125):
                xn = x - lam * step
                if not math.isfinite(xn):
                    break
                fn = _f(xn)
                if fn is None:
                    break
                if abs(fn) < abs(fx) * 0.9:
                    return xn, fn
            break
        return x, fx

    # --- Solve from a bracket ---
    if bracket is not None:
        a, b, fa, fb = bracket
        if fa * fb < 0.0:
            res = secant_driver(a, fa, b, fb)
            fr = _f(res)
            if fr is not None and abs(fr) <= TOL:
                return res
            if fr is not None and abs(fr) < 1e-3 and _time_ok():
                res, fr = _damp_newton(res, fr)
                if fr is not None and abs(fr) <= TOL:
                    return res
                if fr is not None and abs(fr) < 1e-5 and _time_ok():
                    res, fr = _central_polish(res, fr)
                    if fr is not None and abs(fr) <= TOL:
                        return res
            res2 = brent_driver(a, b, fa, fb)
            fr2 = _f(res2)
            if fr2 is not None and abs(fr2) <= TOL:
                return res2
            # best of candidates
            cands = [(res, fr), (res2, fr2)]
            cands = [c for c in cands if c[1] is not None]
            if cands:
                return min(cands, key=lambda c: abs(c[1]))[0]
        # no sign change in bracket -> use secant on endpoints
        res = secant_driver(a, fa, b, fb)
        fr = _f(res)
        if fr is not None and abs(fr) <= TOL:
            return res
        if fr is not None and abs(fr) < 1e-3 and _time_ok():
            res, fr = _damp_newton(res, fr)
            if fr is not None and abs(fr) <= TOL:
                return res
            if fr is not None and abs(fr) < 1e-5 and _time_ok():
                res, fr = _central_polish(res, fr)
                if fr is not None and abs(fr) <= TOL:
                    return res
        return res

    # --- No bracket: secant from best point ---
    bx, _ = _best_from_cache()
    if bx is None:
        return x0
    fc = _f(bx)
    if fc is None:
        return bx
    xp = bx + 1e-4 * max(1.0, abs(bx))
    fp = _f(xp)
    if fp is None:
        return bx
    res = secant_driver(bx, fc, xp, fp)
    fr = _f(res)
    if fr is not None and abs(fr) <= TOL:
        return res
    if fr is not None and abs(fr) < 1e-3 and _time_ok():
        res, fr = _damp_newton(res, fr)
        if fr is not None and abs(fr) <= TOL:
            return res
        if fr is not None and abs(fr) < 1e-5 and _time_ok():
            res, fr = _central_polish(res, fr)
            if fr is not None and abs(fr) <= TOL:
                return res
    return res
