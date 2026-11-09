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

    # Build adaptive sample set with strategic points
    points = [(x0, f0)]

    # Evaluate a small set of initial points to build interpolation
    # Use a geometric spread around x0
    initial_scale = 1.0
    for scale in [1.0, 0.5, 2.0, 4.0, 8.0, 16.0]:
        for sign in [1.0, -1.0]:
            px = x0 + sign * scale
            if not np.isfinite(px):
                continue
            if any(abs(px - ex) <= 1e-14 * max(1.0, abs(px), abs(ex)) for ex, _ in points):
                continue
            fx = get_f(px)
            points.append((px, fx))
            update_best(px, fx)
            if np.isfinite(fx) and abs(fx) <= 1e-8:
                return px
            if call_count[0] > 15:
                break
        if call_count[0] > 15:
            break

    # Iteratively refine using polynomial interpolation
    max_iters = 10
    for iteration in range(max_iters):
        if call_count[0] > 40:
            break

        # Fit polynomial to finite points
        xs = np.array([p[0] for p in points], dtype=float)
        ys = np.array([p[1] for p in points], dtype=float)
        finite = np.isfinite(ys)
        xs_f = xs[finite]
        ys_f = ys[finite]

        candidates = []
        if len(xs_f) >= 2:
            # Use degree up to min(len-1, 4) but prefer lower degree for stability
            deg = min(len(xs_f) - 1, 3)
            try:
                coeffs = np.polyfit(xs_f, ys_f, deg)
                roots = np.roots(coeffs)
                for r in roots:
                    if np.isreal(r):
                        rr = float(np.real(r))
                        if np.isfinite(rr) and abs(rr) < 1e10:
                            # Check if this root is within or near the range of our points
                            if abs(rr - np.mean(xs_f)) < 100 * max(1.0, np.std(xs_f)):
                                candidates.append(rr)
            except Exception:
                candidates = []

        # Add sign-change midpoints
        spts = sorted(points, key=lambda t: t[0])
        for i in range(len(spts) - 1):
            xa, fa = spts[i]
            xb, fb = spts[i + 1]
            if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0:
                # Linear interpolation for better estimate
                if fb != fa:
                    t = fa / (fa - fb)
                    mid = xa + t * (xb - xa)
                else:
                    mid = 0.5 * (xa + xb)
                candidates.append(mid)

        if not candidates:
            # Expand search range
            scale = max(1.0, 2.0 * max(abs(p[0] - x0) for p in points) if points else 1.0)
            for sign in [1.0, -1.0]:
                px = x0 + sign * scale
                if not np.isfinite(px):
                    continue
                if any(abs(px - ex) <= 1e-14 * max(1.0, abs(px), abs(ex)) for ex, _ in points):
                    continue
                fx = get_f(px)
                points.append((px, fx))
                update_best(px, fx)
                if np.isfinite(fx) and abs(fx) <= 1e-8:
                    return px
            continue

        # Evaluate top candidates closest to current best
        candidates = sorted(candidates, key=lambda c: abs(c - best_x))
        for c in candidates[:3]:
            fc = get_f(c)
            update_best(c, fc)
            if np.isfinite(fc) and abs(fc) <= 1e-8:
                return c
            # Add this point to our set for next iteration
            points.append((c, fc))
            if call_count[0] > 45:
                break

        if best_abs is not None and best_abs <= 1e-8:
            return best_x

    # Final polish with secant method
    x = best_x
    fx = get_f(x)
    if not np.isfinite(fx):
        return best_x

    # Find a second point for secant
    h = 1e-4 * max(1.0, abs(x))
    x2 = x + h
    fx2 = get_f(x2)

    for _ in range(6):
        if not np.isfinite(fx) or not np.isfinite(fx2) or abs(fx) <= 1e-12:
            break
        if fx2 == fx:
            break
        # Secant step
        step = fx * (x - x2) / (fx - fx2)
        if not np.isfinite(step) or abs(step) > 100.0 * max(1.0, abs(x)):
            break
        xn = x - step
        if not np.isfinite(xn):
            break
        fxn = get_f(xn)
        if not np.isfinite(fxn):
            break
        if abs(fxn) < abs(fx):
            x, fx = xn, fxn
            x2, fx2 = x, fx
        else:
            x2, fx2 = xn, fxn
        if abs(fx) <= 1e-12:
            break
        if call_count[0] > 50:
            break

    update_best(x, fx)
    return best_x
# EVOLVE-BLOCK-END
