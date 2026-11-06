"""Root finding: hybrid secant/Newton + bisection + derivative-free Nelder-Mead."""
# EVOLVE-BLOCK-START
import time
import math


def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 140.0   # safety margin under the 150s wall-clock limit
    max_evals = 480                 # safety margin under the 500-eval limit
    state = {'count': 0}

    def evalf(x):
        if state['count'] >= max_evals:
            return None
        state['count'] += 1
        try:
            v = f(x)
        except Exception:
            return None
        if v is None:
            return None
        try:
            v = float(v)
        except Exception:
            return None
        if not math.isfinite(v):
            return None
        return v

    def timed():
        return time.time() >= deadline

    def absf(x):
        v = evalf(x)
        return abs(v) if v is not None else float('inf')

    x0 = float(x0)
    best = x0
    best_abs = absf(x0)
    if best_abs <= 1e-8:
        return best

    def update(x):
        nonlocal best, best_abs
        if x is None or not math.isfinite(x):
            return
        a = absf(x)
        if a < best_abs:
            best_abs = a
            best = x
            if a <= 1e-8:
                return True
        return False

    # ---------- Phase 1: Secant (1 eval/iter) ----------
    a = x0
    fa = evalf(a)
    if fa is None:
        return best
    if abs(fa) <= 1e-8:
        return a
    h = 1e-6 * max(1.0, abs(a))
    b = a + h
    fb = evalf(b)
    if fb is None:
        fb = fa
    if fb == fa:
        b = a + 1e-3 * max(1.0, abs(a))
        fb = evalf(b)
        if fb is None:
            fb = fa

    for _ in range(120):
        if timed():
            break
        if fb == fa:
            break
        denom = (fb - fa) / (b - a)
        if denom == 0 or not math.isfinite(denom):
            break
        xn = a - fa / denom
        if not math.isfinite(xn):
            break
        # damping for stability
        step = xn - a
        if abs(step) > 10.0 * max(1.0, abs(a)):
            xn = a + math.copysign(10.0 * max(1.0, abs(a)), step)
        fxn = evalf(xn)
        if fxn is None:
            break
        if abs(fxn) <= 1e-8:
            return xn
        a, fa = b, fb
        b, fb = xn, fxn
        update(xn)
        if timed():
            break

    # ---------- Phase 2: Central-diff Newton (3 evals/iter) ----------
    x = best
    fx = evalf(x)
    if fx is None:
        return best
    if abs(fx) <= 1e-8:
        return x
    for _ in range(60):
        if timed():
            break
        h = 1e-6 * max(1.0, abs(x))
        fph = evalf(x + h)
        fmh = evalf(x - h)
        if fph is None or fmh is None:
            break
        d = (fph - fmh) / (2.0 * h)
        if d == 0 or not math.isfinite(d):
            break
        xn = x - fx / d
        step = xn - x
        if abs(step) > 10.0 * max(1.0, abs(x)):
            xn = x + math.copysign(10.0 * max(1.0, abs(x)), step)
        fxn = evalf(xn)
        if fxn is None:
            break
        if abs(fxn) <= 1e-8:
            return xn
        x, fx = xn, fxn
        update(xn)

    # ---------- Phase 3: Bracket + Bisection ----------
    lo, hi, flo, fhi = None, None, None, None
    # reuse last known points to seed bracket
    cand = [x, best, a, b, x0]
    seen = set()
    vals = []
    for c in cand:
        if c is None or not math.isfinite(c):
            continue
        key = round(c, 12)
        if key in seen:
            continue
        seen.add(key)
        v = evalf(c)
        if v is not None:
            vals.append((c, v))
    if len(vals) >= 2:
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                (x1, v1), (x2, v2) = vals[i], vals[j]
                if v1 == 0:
                    return x1
                if v2 == 0:
                    return x2
                if v1 * v2 < 0:
                    lo, hi, flo, fhi = x1, x2, v1, v2
                    break
            if lo is not None:
                break
    # if no bracket found, expand around best
    if lo is None:
        center = best
        scale = 1e-6 * max(1.0, abs(center))
        lo, hi = center - scale, center + scale
        flo = evalf(lo)
        fhi = evalf(hi)
        for _ in range(30):
            if timed():
                break
            if flo is None or fhi is None:
                break
            if flo * fhi < 0:
                break
            if flo == 0:
                return lo
            if fhi == 0:
                return hi
            scale *= 2.0
            lo, hi = center - scale, center + scale
            flo = evalf(lo)
            fhi = evalf(hi)
    # bisection
    if lo is not None and hi is not None and flo is not None and fhi is not None:
        if flo * fhi >= 0:
            lo = hi = None
    if lo is not None:
        for _ in range(80):
            if timed():
                break
            mid = 0.5 * (lo + hi)
            if mid == lo or mid == hi:
                break
            fm = evalf(mid)
            if fm is None:
                break
            if abs(fm) <= 1e-8:
                return mid
            update(mid)
            if flo * fm <= 0:
                hi, fhi = mid, fm
            else:
                lo, flo = mid, fm
            if hi - lo <= 1e-12 * max(1.0, abs(mid)):
                break

    # ---------- Phase 4: Nelder-Mead on |f(x)|^2 (derivative-free) ----------
    if best_abs > 1e-8:
        x = best
        scale = 1e-3 * max(1.0, abs(x))
        p0 = x
        p1 = x + scale
        p2 = x - scale
        v0 = absf(p0) ** 2
        v1 = absf(p1) ** 2
        v2 = absf(p2) ** 2
        pts = [(p0, v0), (p1, v1), (p2, v2)]
        for _ in range(150):
            if timed():
                break
            # sort
            pts.sort(key=lambda t: t[1])
            if pts[0][1] <= 1e-16:  # |f|^2 <= 1e-16 -> |f| <= 1e-8
                return pts[0][0]
            worst = pts[2]
            # centroid of two best
            cx = 0.5 * (pts[0][0] + pts[1][0])
            # reflection
            xr = cx + (cx - worst[0])
            vr = absf(xr) ** 2
            if vr < pts[1][1]:
                # expansion
                xe = cx + 2.0 * (cx - worst[0])
                ve = absf(xe) ** 2
                if ve < vr:
                    pts[2] = (xe, ve)
                else:
                    pts[2] = (xr, vr)
            else:
                # contraction
                xc = worst[0] + 0.5 * (cx - worst[0])
                vc = absf(xc) ** 2
                if vc < worst[1]:
                    pts[2] = (xc, vc)
                else:
                    # shrink toward best
                    base = pts[0][0]
                    for i in range(3):
                        p, v = pts[i]
                        np_ = base + 0.5 * (p - base)
                        nv = absf(np_) ** 2
                        pts[i] = (np_, nv)
            update(pts[0][0])
            if timed():
                break

    if best_abs <= 1e-8:
        return best
    return best
# EVOLVE-BLOCK-END
