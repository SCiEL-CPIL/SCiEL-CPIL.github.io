"""Polynomial-interpolation root finder for 3 fixed smooth polynomial equations."""
# EVOLVE-BLOCK-START
import numpy as np


def find_root(f, x0):
    x0 = float(x0)
    cache = {}
    call_count = [0]
    best_x = x0
    best_abs = None

    def get_f(x):
        x = float(x)
        if not np.isfinite(x):
            return np.inf
        key = x
        if key in cache:
            return cache[key]
        call_count[0] += 1
        try:
            val = float(f(x))
        except Exception:
            val = np.inf
        cache[key] = val
        return val

    def update_best(x, fx):
        nonlocal best_x, best_abs
        if not np.isfinite(x) or not np.isfinite(fx):
            return
        a = abs(fx)
        if best_abs is None or a < best_abs:
            best_abs = a
            best_x = x

    # Initial evaluation at x0
    f0 = get_f(x0)
    update_best(x0, f0)
    if np.isfinite(f0) and abs(f0) <= 1e-8:
        return x0

    # Build adaptive sample set
    points = [(x0, f0)]

    # Candidate offsets (relative to a running scale)
    def make_points(scale):
        offs = [0.0, 1.0, -1.0, 2.5, -2.5, 6.0, -6.0, 15.0, -15.0]
        pts = []
        for o in offs:
            px = x0 + o * scale
            if np.isfinite(px):
                pts.append(px)
        return pts

    scale = 1.0
    max_scale = 1e6
    expansions = 0
    max_expansions = 8

    while True:
        # Evaluate new points up to a cap
        target_pts = make_points(scale)
        for px in target_pts:
            if len(points) >= 12:
                break
            if any(abs(px - ex) <= 1e-14 * max(1.0, abs(px), abs(ex)) for ex, _ in points):
                continue
            fx = get_f(px)
            points.append((px, fx))
            update_best(px, fx)
            if np.isfinite(fx) and abs(fx) <= 1e-8:
                return px

        # Fit polynomial
        xs = np.array([p[0] for p in points], dtype=float)
        ys = np.array([p[1] for p in points], dtype=float)
        finite = np.isfinite(ys)
        xs_f = xs[finite]
        ys_f = ys[finite]

        candidates = []
        if len(xs_f) >= 2:
            deg = min(len(xs_f) - 1, 4)
            try:
                coeffs = np.polyfit(xs_f, ys_f, deg)
                roots = np.roots(coeffs)
                for r in roots:
                    if np.isreal(r):
                        rr = float(np.real(r))
                        if np.isfinite(rr) and abs(rr) < 1e12:
                            candidates.append(rr)
            except Exception:
                candidates = []

        # Also include sign-change midpoints as candidates
        spts = sorted(points, key=lambda t: t[0])
        for i in range(len(spts) - 1):
            xa, fa = spts[i]
            xb, fb = spts[i + 1]
            if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0:
                candidates.append(0.5 * (xa + xb))

        # Evaluate a few best candidates (closest to x0 / smallest span)
        if candidates:
            # score candidates by distance to x0
            candidates = sorted(candidates, key=lambda c: abs(c - x0))
            for c in candidates[:4]:
                fc = get_f(c)
                update_best(c, fc)
                if np.isfinite(fc) and abs(fc) <= 1e-8:
                    return c

        if best_abs is not None and best_abs <= 1e-8:
            return best_x

        expansions += 1
        if expansions >= max_expansions or scale >= max_scale:
            break
        scale *= 5.0
        if call_count[0] > 30:
            break

    # Polish best_x with a few Newton/secant steps using finite-difference derivative
    x = best_x
    fx = get_f(x)
    for _ in range(4):
        if not np.isfinite(fx) or abs(fx) <= 1e-12:
            break
        h = 1e-6 * max(1.0, abs(x))
        d = (get_f(x + h) - fx) / h
        if not np.isfinite(d) or d == 0:
            break
        step = fx / d
        # damp step to avoid flying off
        if abs(step) > 10.0 * max(1.0, abs(x)):
            step = np.sign(step) * 10.0 * max(1.0, abs(x))
        xn = x - step
        if not np.isfinite(xn):
            break
        fxn = get_f(xn)
        if np.isfinite(fxn) and abs(fxn) < abs(fx):
            x, fx = xn, fxn
        else:
            # secant fallback using last two
            break
        if abs(fx) <= 1e-12:
            break
        if call_count[0] > 40:
            break
    update_best(x, fx)
    return best_x
# EVOLVE-BLOCK-END
