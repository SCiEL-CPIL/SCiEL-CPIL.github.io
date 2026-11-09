"""Root finder for 3 fixed oscillatory equations."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f (oscillatory) starting near x0.
    Strategy:
      1. Evaluate f at x0; if good enough, return.
      2. Geometric scan with 8 scales to find sign-change brackets.
      3. If no bracket found, do a finer secondary local scan.
      4. Refine best bracket with bisection + secant acceleration.
      5. Fallback: secant from best two points.
      6. Deadline guard: return best-so-far before wall-clock limit.
    """
    deadline = time.time() + 148.0  # safety margin under 150s
    tol = 1e-8
    x0 = float(x0)

    # Cache of evaluated points: x -> f(x)
    cache = {}

    def get_f(xv):
        """Evaluate f at xv, using cache if available."""
        if xv in cache:
            return cache[xv]
        try:
            fv = f(xv)
        except Exception:
            return None
        if not math.isfinite(fv):
            return None
        cache[xv] = fv
        return fv

    # Track global best (smallest |f|)
    best_x = x0
    best_f = get_f(x0)
    if best_f is None:
        best_f = float('inf')
    best_abs = abs(best_f)

    if best_abs <= tol:
        return best_x

    def update_best(xv, fv):
        nonlocal best_x, best_f, best_abs
        if not math.isfinite(xv) or not math.isfinite(fv):
            return
        af = abs(fv)
        if af < best_abs:
            best_x = xv
            best_f = fv
            best_abs = af

    # ---- Phase 1: Geometric scan to find sign-change brackets ----
    # 8 scales in both directions = 16 evaluations
    base_offsets = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]
    brackets = []  # list of (a, b, fa, fb)
    eval_order = []  # track (x, f) pairs in evaluation order

    for s in base_offsets:
        for sign in [1, -1]:
            xv = x0 + sign * s
            fv = get_f(xv)
            if fv is None:
                continue
            eval_order.append((xv, fv))
            update_best(xv, fv)
            if best_abs <= tol:
                return best_x
            # Check bracket with x0
            fx0 = cache.get(x0)
            if fx0 is not None and fx0 * fv < 0:
                lo, hi = min(x0, xv), max(x0, xv)
                brackets.append((lo, hi, cache[lo], cache[hi]))
            # Check bracket with all previously evaluated points
            for (px, pf) in eval_order[:-1]:
                if pf * fv < 0:
                    lo, hi = min(px, xv), max(px, xv)
                    # Avoid duplicate brackets
                    dup = False
                    for (ba, bb, _, _) in brackets:
                        if abs(ba - lo) < 1e-12 and abs(bb - hi) < 1e-12:
                            dup = True
                            break
                    if not dup:
                        brackets.append((lo, hi, pf, fv))

    # ---- Phase 1b: If no bracket found, do finer secondary local scan ----
    if not brackets:
        fine_offsets = [1e-5, 5e-5, 2e-4, 5e-4, 2e-3, 5e-3, 2e-2, 5e-2, 0.5]
        for s in fine_offsets:
            if time.time() > deadline:
                break
            for sign in [1, -1]:
                if time.time() > deadline:
                    break
                xv = x0 + sign * s
                if xv in cache:
                    fv = cache[xv]
                else:
                    fv = get_f(xv)
                if fv is None:
                    continue
                update_best(xv, fv)
                if best_abs <= tol:
                    return best_x
                # Check bracket with all previously evaluated points
                for (px, pf) in eval_order:
                    if pf * fv < 0:
                        lo, hi = min(px, xv), max(px, xv)
                        brackets.append((lo, hi, pf, fv))
                        break  # one bracket is enough to start refining
                if brackets:
                    break
            if brackets:
                break

    # ---- Phase 2: Refine brackets with bisection + secant acceleration ----
    # Sort brackets by span (tighter first)
    brackets.sort(key=lambda t: abs(t[1] - t[0]))

    for (a, b, fa, fb) in brackets:
        if time.time() > deadline:
            break

        xa, xb = a, b
        fa, fb = fa, fb

        for _ in range(100):
            if time.time() > deadline:
                break

            if abs(fa) <= tol:
                update_best(xa, fa)
                return xa
            if abs(fb) <= tol:
                update_best(xb, fb)
                return xb

            # Try secant step if it stays within the bracket
            denom = fb - fa
            if abs(denom) > 1e-15:
                xs = xb - fb * (xb - xa) / denom
                if xa < xs < xb:  # secant step is inside bracket
                    fs = get_f(xs)
                    if fs is not None:
                        update_best(xs, fs)
                        if abs(fs) <= tol:
                            return xs
                        if fa * fs <= 0:
                            xb, fb = xs, fs
                        else:
                            xa, fa = xs, fs
                        continue

            # Bisection step
            xm = 0.5 * (xa + xb)
            if xm == xa or xm == xb:
                break
            fm = get_f(xm)
            if fm is None:
                break
            update_best(xm, fm)
            if abs(fm) <= tol:
                return xm
            if fa * fm <= 0:
                xb, fb = xm, fm
            else:
                xa, fa = xm, fm

        # After refinement, return the better endpoint
        if abs(fa) <= abs(fb):
            update_best(xa, fa)
            return xa
        else:
            update_best(xb, fb)
            return xb

    # ---- Phase 3: No bracket found — use secant from best two points ----
    if len(cache) >= 2:
        sorted_pts = sorted(cache.items(), key=lambda kv: abs(kv[1]))
        (x1, f1), (x2, f2) = sorted_pts[0], sorted_pts[1]
        if x1 != x2 and abs(f1 - f2) > 1e-15:
            xa, xb = x1, x2
            fa, fb = f1, f2
            for _ in range(80):
                if time.time() > deadline:
                    break
                if abs(fa) <= tol:
                    return xa
                if abs(fb) <= tol:
                    return xb
                denom = fb - fa
                if abs(denom) < 1e-15:
                    break
                xn = xb - fb * (xb - xa) / denom
                if not math.isfinite(xn):
                    break
                fn = get_f(xn)
                if fn is None:
                    break
                update_best(xn, fn)
                if abs(fn) <= tol:
                    return xn
                xa, fa = xb, fb
                xb, fb = xn, fn

    return best_x
# EVOLVE-BLOCK-END
