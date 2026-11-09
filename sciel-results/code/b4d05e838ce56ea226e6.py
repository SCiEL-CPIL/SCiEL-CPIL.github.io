"""Root finding for ill-conditioned equations: secant + bracket + NM + polish."""
import time
import math

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8, using at most 500 evaluations of f.
    Returns the best x found (smallest |f(x)|) within the budget.
    """
    t0 = time.time()
    eval_count = 0
    max_evals = 500
    tol = 1e-8
    wall_limit = 144.0

    best_x = float(x0)
    try:
        best_fx = float(f(best_x))
        eval_count += 1
        best_abs_fx = abs(best_fx) if math.isfinite(best_fx) else float('inf')
    except Exception:
        best_fx = 0.0
        best_abs_fx = float('inf')

    def _time_left():
        return wall_limit - (time.time() - t0)

    def _eval(x):
        nonlocal eval_count, best_x, best_fx, best_abs_fx
        if eval_count >= max_evals or _time_left() <= 0.3:
            return None, False
        try:
            fx = float(f(x))
        except Exception:
            return None, False
        eval_count += 1
        if not math.isfinite(fx):
            return None, False
        ax = abs(fx)
        if ax < best_abs_fx:
            best_x = x
            best_fx = fx
            best_abs_fx = ax
        return fx, True

    def _done():
        return best_abs_fx <= tol

    if _done():
        return best_x

    # ---- Phase 1: adaptive secant with safeguarded step ----
    x1 = float(x0)
    f1 = best_fx
    h0 = 1e-4 * max(1.0, abs(x1))
    x2 = x1 + h0
    f2, ok = _eval(x2)
    if not ok:
        x2 = x1 - h0
        f2, ok = _eval(x2)
    if not ok:
        return best_x

    prev_bad = 0
    MAX_BAD = 8
    for _ in range(180):
        if _done():
            return best_x
        if eval_count >= max_evals or _time_left() < 1.5:
            break
        dx = x2 - x1
        if abs(dx) < 1e-15:
            break
        d = (f2 - f1) / dx
        if abs(d) < 1e-15:
            break
        xn = x2 - f2 / d
        cap = 10.0 * max(1.0, abs(x2))
        if abs(xn - x2) > cap:
            xn = x2 - math.copysign(cap, xn - x2)
        fn, ok = _eval(xn)
        if not ok:
            break
        if abs(fn) <= tol:
            return xn
        if abs(fn) < abs(f2) * 0.95:
            prev_bad = 0
        else:
            prev_bad += 1
            if prev_bad >= MAX_BAD:
                break
        x1, f1 = x2, f2
        x2, f2 = xn, fn

    if _done():
        return best_x

    # ---- Phase 2: bracket search + bisection/IQU ----
    a = x2
    fa = f2
    if fa is None:
        fa, _ = _eval(a)
        if fa is None:
            return best_x

    b = fb = None
    scale = 1.0
    for _ in range(25):
        if _time_left() < 1.5 or eval_count >= max_evals - 10:
            break
        s = max(1.0, abs(a))
        bp = a + scale * s
        fbp, ok1 = _eval(bp)
        if ok1 and fa * fbp < 0:
            b, fb = bp, fbp
            break
        bn = a - scale * s
        fbn, ok2 = _eval(bn)
        if ok2 and fa * fbn < 0:
            b, fb = bn, fbn
            break
        scale *= 1.6

    if b is not None and fb is not None and fa * fb < 0:
        lo, hi = min(a, b), max(a, b)
        flo = fa if a == lo else fb
        fhi = fa if a == hi else fb
        if flo * fhi > 0:
            lo, hi = hi, lo
            flo, fhi = fhi, flo
        for _ in range(70):
            if _done():
                return best_x
            if eval_count >= max_evals or _time_left() < 1.0:
                break
            if abs(flo) <= tol:
                return lo
            if abs(fhi) <= tol:
                return hi
            # inverse quadratic interpolation
            s = None
            den = fhi * (flo + fhi) + flo * (fhi - flo)
            if abs(den) > 1e-30:
                s = lo + (hi - lo) * flo * (flo - fhi) / den
                if not math.isfinite(s) or s <= lo or s >= hi:
                    s = None
            if s is None:
                s = 0.5 * (lo + hi)
            fs, ok = _eval(s)
            if not ok:
                break
            if abs(fs) <= tol:
                return s
            if flo * fs < 0:
                hi, fhi = s, fs
            else:
                lo, flo = s, fs

    if _done():
        return best_x

    # ---- Phase 3: Nelder-Mead on g(x) = f(x)^2 ----
    def _g(x):
        fx, ok = _eval(x)
        if not ok:
            return None, False
        return fx * fx, True

    sc = 1e-3 * max(1.0, abs(best_x))
    simplex = [best_x, best_x + sc, best_x - sc]
    gvals = []
    ok_sx = True
    for xi in simplex:
        gi, ok = _g(xi)
        if not ok:
            ok_sx = False
            break
        gvals.append(gi)
        if gi <= tol * tol:
            return xi

    if ok_sx:
        alpha, gamma, rho, sigma = 1.5, 2.5, 0.4, 0.4
        for _ in range(70):
            if _done():
                return best_x
            if eval_count >= max_evals - 3 or _time_left() < 1.0:
                break
            order = sorted(range(3), key=lambda i: gvals[i])
            simplex = [simplex[i] for i in order]
            gvals = [gvals[i] for i in order]
            if gvals[0] <= tol * tol:
                return simplex[0]
            if abs(gvals[2] - gvals[0]) < 1e-18:
                break
            c = 0.5 * (simplex[0] + simplex[1])
            xr = c + alpha * (c - simplex[2])
            gr, ok = _g(xr)
            if not ok:
                break
            if gr <= gvals[0]:
                xe = c + gamma * (xr - c)
                ge, ok2 = _g(xe)
                if ok2 and ge < gr:
                    simplex[2], gvals[2] = xe, ge
                else:
                    simplex[2], gvals[2] = xr, gr
            elif gr < gvals[1]:
                simplex[2], gvals[2] = xr, gr
            else:
                xc = c + rho * (simplex[2] - c)
                gc, ok3 = _g(xc)
                if ok3 and gc < gvals[2]:
                    simplex[2], gvals[2] = xc, gc
                else:
                    for i in range(2):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        gi, ok4 = _g(simplex[i])
                        if not ok4:
                            break
                        gvals[i] = gi
                    else:
                        continue
                    break

    if _done():
        return best_x

    # ---- Phase 4: multi-start secant from diverse offsets ----
    ls = max(1.0, abs(best_x))
    if abs(x2 - x1) > 1e-15:
        lsp = abs(x2 - x1)
        if lsp < 1e-10:
            lsp = 1e-4 * ls
    else:
        lsp = 1e-4 * ls

    offs = [1e-3, -1e-3, 1e-2, -1e-2, 0.1, -0.1, 1.0, -1.0, 5.0, -5.0,
            0.01, -0.01, 0.5, -0.5, 10.0, -10.0,
            lsp, -lsp, 2*lsp, -2*lsp, 0.5*lsp, -0.5*lsp]
    seen = set()
    uoffs = []
    for o in offs:
        k = round(o, 12)
        if k not in seen and abs(o) > 1e-15:
            seen.add(k)
            uoffs.append(o)

    for o in uoffs:
        if eval_count >= max_evals - 15 or _time_left() < 1.5:
            break
        xt = float(x0) + o * max(1.0, abs(x0))
        ft, ok = _eval(xt)
        if not ok:
            continue
        if abs(ft) <= tol:
            return xt
        hr = 1e-4 * max(1.0, abs(xt))
        x2r = xt + hr
        f2r, ok2 = _eval(x2r)
        if not ok2:
            x2r = xt - hr
            f2r, ok2 = _eval(x2r)
        if not ok2:
            continue
        bad_r = 0
        for _ in range(12):
            if _done():
                return best_x
            if eval_count >= max_evals or _time_left() < 1.0:
                break
            dxr = x2r - xt
            if abs(dxr) < 1e-15:
                break
            dr = (f2r - ft) / dxr
            if abs(dr) < 1e-15:
                break
            xn = x2r - f2r / dr
            fn, ok3 = _eval(xn)
            if not ok3:
                break
            if abs(fn) <= tol:
                return xn
            if abs(fn) < abs(f2r) * 0.95:
                bad_r = 0
            else:
                bad_r += 1
                if bad_r >= 4:
                    break
            xt, ft = x2r, f2r
            x2r, f2r = xn, fn

    if _done():
        return best_x

    # ---- Phase 5: high-precision polish (IQU + Newton) ----
    if eval_count < max_evals - 8 and _time_left() > 1.5 and best_abs_fx < 1e-2:
        hp = 1e-6 * max(1.0, abs(best_x))
        xa, xb = best_x - hp, best_x + hp
        fa2, oka = _eval(xa)
        fb2, okb = _eval(xb)
        if oka and okb and abs(fa2) < 1.0 and abs(fb2) < 1.0:
            ya, yb, yc = fa2, best_fx, fb2
            xa2, xb2, xc2 = xa, best_x, xb
            for _ in range(35):
                if _done():
                    return best_x
                if eval_count >= max_evals or _time_left() < 0.8:
                    break
                den = ya*yb*(ya-yb) + yb*yc*(yb-yc) + yc*ya*(yc-ya)
                if abs(den) < 1e-30:
                    break
                num = xa2*xa2*yb*yc*(yb-yc) + xb2*xb2*ya*yc*(yc-ya) + xc2*xc2*ya*yb*(ya-yb)
                xn = num / den
                if not math.isfinite(xn):
                    break
                fn, ok = _eval(xn)
                if not ok:
                    break
                if abs(fn) <= tol:
                    return xn
                # replace worst
                if abs(fn) < abs(ya):
                    xa2, ya = xn, fn
                elif abs(fn) < abs(yb):
                    xb2, yb = xn, fn
                else:
                    xc2, yc = xn, fn

        # Newton polish with finite-difference derivative
        if not _done() and eval_count < max_evals - 3 and _time_left() > 1.0:
            xs = best_x
            fs = best_fx
            hnd = 1e-10 * max(1.0, abs(xs))
            xp = xs + hnd
            fp, okp = _eval(xp)
            if okp:
                d_est = (fp - fs) / hnd
            else:
                d_est = None
            for _ in range(40):
                if _done():
                    return best_x
                if eval_count >= max_evals or _time_left() < 0.5:
                    break
                if d_est is not None and abs(d_est) > 1e-15:
                    xn = xs - fs / d_est
                else:
                    break
                if not math.isfinite(xn):
                    break
                fn, ok = _eval(xn)
                if not ok:
                    break
                if abs(fn) <= tol:
                    return xn
                ddx = xn - xs
                if abs(ddx) > 1e-15:
                    d_est = (fn - fs) / ddx
                if abs(fn) < abs(fs) * 0.999:
                    xs, fs = xn, fn
                else:
                    break

    return best_x
