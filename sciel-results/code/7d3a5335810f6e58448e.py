"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    DEADLINE = time.time() + 110.0  # safety margin before 120s
    MAX_CALLS = 480
    calls = [0]

    def feval(x):
        if calls[0] >= MAX_CALLS:
            return None
        calls[0] += 1
        fx = f(x)
        if fx is None or not math.isfinite(fx):
            return None
        return fx

    x = float(x0)
    if not math.isfinite(x):
        x = 0.0

    fx = feval(x)
    if fx is None:
        return x
    if abs(fx) <= 1e-8:
        return x

    best_x = x
    best_abs = abs(fx)

    # Store points for interpolation: list of (x, f(x))
    pts = [(x, fx)]

    def add_pt(xv, fv):
        if fv is None:
            return
        # avoid duplicates
        for (px, pf) in pts:
            if abs(px - xv) < 1e-15 * max(1.0, abs(xv)):
                return
        pts.append((xv, fv))

    def check_and_update(xv, fv):
        nonlocal best_x, best_abs
        if fv is None:
            return
        av = abs(fv)
        if av < best_abs:
            best_abs = av
            best_x = xv

    # Phase 1: Build initial bracket and a few points via expansion
    # Try to find a sign change bracket [a, b]
    a = b = None
    fa = fb = None
    step = 1e-2 * max(1.0, abs(x))
    for i in range(30):
        if time.time() > DEADLINE:
            return best_x
        s = step * (2 ** i)
        left = x - s
        right = x + s
        fl = feval(left)
        fr = feval(right)
        if fl is None or fr is None:
            continue
        add_pt(left, fl)
        add_pt(right, fr)
        check_and_update(left, fl)
        check_and_update(right, fr)
        if abs(fl) <= 1e-8:
            return left
        if abs(fr) <= 1e-8:
            return right
        if fl * fr < 0:
            a, b = left, right
            fa, fb = fl, fr
            break
        # If no sign change yet, also record the best
        if abs(fl) < abs(fx):
            a, b = left, x
            fa, fb = fl, fx
        elif abs(fr) < abs(fx):
            a, b = x, right
            fa, fb = fx, fr

    # Phase 2: Iterative refinement
    # Use interpolation-based methods when we have enough points,
    # fall back to bisection if we have a bracket.
    # We'll maintain a window of recent points for interpolation.
    # Strategy: use cubic interpolation on the 4 most recent distinct points
    # when they span a bracket; otherwise use inverse quadratic interpolation
    # or secant, with bisection fallback.

    # Ensure we have at least a bracket or at least 2 points
    if a is not None and b is not None:
        # We have a bracket [a, b] with sign change (or best-so-far)
        pass
    elif len(pts) >= 2:
        # Use the two points with smallest |f| as a pseudo-bracket for secant
        pts_sorted = sorted(pts, key=lambda p: abs(p[1]))
        p1, p2 = pts_sorted[0], pts_sorted[1]
        a, b = p1[0], p2[0]
        fa, fb = p1[1], p2[1]
    else:
        # Only one point; do a simple secant-like step
        h = 1e-4 * max(1.0, abs(x))
        x2 = x + h
        fx2 = feval(x2)
        if fx2 is None:
            return best_x
        add_pt(x2, fx2)
        check_and_update(x2, fx2)
        if abs(fx2) <= 1e-8:
            return x2
        if fx * fx2 < 0:
            a, b = x, x2
            fa, fb = fx, fx2
        else:
            # Move toward the smaller value
            if abs(fx2) < abs(fx):
                x = x2
                fx = fx2
            h = -h
            x2 = x + h
            fx2 = feval(x2)
            if fx2 is None:
                return best_x
            add_pt(x2, fx2)
            check_and_update(x2, fx2)
            if abs(fx2) <= 1e-8:
                return x2
            a, b = x, x2
            fa, fb = fx, fx2

    # Main iteration loop
    # Keep a history of the last few (x, f) pairs for interpolation
    # We'll use a list `hist` of the most recent points, up to 5
    hist = []
    # Seed hist with the bracket endpoints and x0 if present
    for (px, pf) in pts:
        hist.append((px, pf))
    hist.sort()
    # Limit hist to most recent 5 by x
    if len(hist) > 5:
        hist = hist[-5:]

    for it in range(200):
        if time.time() > DEADLINE:
            return best_x
        if calls[0] >= MAX_CALLS - 5:
            break

        # Check convergence
        if best_abs <= 1e-12:
            break

        # Decide next point
        nx = None

        # Try cubic interpolation if we have 4+ points in a bracket
        if a is not None and b is not None and fa * fb < 0:
            # Find 4 points within [a, b] for cubic interpolation
            cand = [p for p in hist if a <= p[0] <= b]
            if len(cand) >= 4:
                # Use the 4 points closest to the current best
                cand.sort(key=lambda p: abs(p[1]))
                # Actually for cubic interp we want 4 distinct x's; use the 4 most recent
                cand4 = hist[-4:]
                # Ensure they are distinct
                if len(set(round(p[0], 15) for p in cand4)) == 4:
                    nx = cubic_interp_root(cand4, a, b)
            if nx is None:
                # Try inverse quadratic interpolation with 3 points
                if len(hist) >= 3:
                    cand3 = hist[-3:]
                    if len(set(round(p[0], 15) for p in cand3)) == 3:
                        nx = iq_interpolation(cand3, a, b)
            if nx is None:
                # Secant method
                if len(hist) >= 2:
                    p1, p2 = hist[-2], hist[-1]
                    if abs(p2[1] - p1[1]) > 1e-30:
                        nx = p2[0] - p2[1] * (p2[0] - p1[0]) / (p2[1] - p1[1])
            if nx is None:
                # Bisection
                nx = 0.5 * (a + b)
        else:
            # No valid bracket; use secant or interpolation on best points
            if len(hist) >= 2:
                p1, p2 = hist[-2], hist[-1]
                if abs(p2[1] - p1[1]) > 1e-30:
                    nx = p2[0] - p2[1] * (p2[0] - p1[0]) / (p2[1] - p1[1])
                else:
                    nx = 0.5 * (p1[0] + p2[0])
            else:
                nx = x + 1e-4 * max(1.0, abs(x))

        # Validate nx
        if not math.isfinite(nx):
            nx = 0.5 * (a + b) if a is not None and b is not None else x + 1e-4

        # Evaluate
        fnx = feval(nx)
        if fnx is None:
            break
        check_and_update(nx, fnx)
        if abs(fnx) <= 1e-8:
            return nx
        add_pt(nx, fnx)

        # Update bracket if sign change
        if a is not None and b is not None and fa * fb < 0:
            if fa * fnx < 0:
                b = nx
                fb = fnx
            elif fb * fnx < 0:
                a = nx
                fa = fnx
            # If no sign change, keep bracket but note we're not bracketing
        else:
            # Try to establish a bracket
            # Compare nx with current best
            if abs(fnx) < best_abs:
                # Move bracket toward nx
                pass

        # Update history
        hist.append((nx, fnx))
        if len(hist) > 5:
            hist.pop(0)

    # Final polish: Newton's method with finite difference derivative
    # Use the best point as starting point
    xp = best_x
    for _ in range(30):
        if time.time() > DEADLINE:
            break
        if calls[0] >= MAX_CALLS:
            break
        fxp = feval(xp)
        if fxp is None:
            break
        if abs(fxp) <= 1e-12:
            break
        h = 1e-7 * max(1.0, abs(xp))
        fxp_h = feval(xp + h)
        if fxp_h is None:
            break
        d = (fxp_h - fxp) / h
        if d == 0 or not math.isfinite(d):
            break
        x_new = xp - fxp / d
        if not math.isfinite(x_new):
            break
        xp = x_new
        check_and_update(xp, fxp)

    return best_x


def cubic_interp_root(points, a, b):
    """
    Fit a cubic polynomial through 4 points and find its root
    in [a, b]. Returns None if no root found in interval.
    """
    if len(points) < 4:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    # Check for duplicate x
    if len(set(round(x, 15) for x in xs)) < 4:
        return None
    # Lagrange interpolation to find coefficients
    # We'll use the factored form to find roots
    # Cubic: p(x) = sum_{i=0}^{3} y_i * L_i(x)
    # Find root by bisection on the interpolant (cheap, no f calls)
    def p(xv):
        s = 0.0
        for i in range(4):
            Li = 1.0
            for j in range(4):
                if j != i:
                    Li *= (xv - xs[j]) / (xs[i] - xs[j])
            s += ys[i] * Li
        return s

    pa = p(a)
    pb = p(b)
    if pa * pb > 0:
        return None
    # Bisection on the interpolant
    lo, hi = a, b
    flo, fhi = pa, pb
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        fm = p(mid)
        if abs(fm) < 1e-15:
            return mid
        if flo * fm < 0:
            hi = mid
            fhi = fm
        else:
            lo = mid
            flo = fm
    return 0.5 * (lo + hi)


def iq_interpolation(points, a, b):
    """
    Inverse quadratic interpolation using 3 points.
    Returns the interpolated x, or None if invalid.
    """
    if len(points) < 3:
        return None
    x0, x1, x2 = points[0][0], points[1][0], points[2][0]
    f0, f1, f2 = points[0][1], points[1][1], points[2][1]
    # Check distinct x
    if abs(x0 - x1) < 1e-15 or abs(x1 - x2) < 1e-15 or abs(x0 - x2) < 1e-15:
        return None
    # Inverse quadratic: fit x as a quadratic in f
    # Using Lagrange on (f, x) pairs
    # x = f0^2 * (f1 - f2) / ((f0 - f1)(f0 - f2)) * ... 
    # Standard IQU formula:
    denom0 = (f0 - f1) * (f0 - f2)
    denom1 = (f1 - f0) * (f1 - f2)
    denom2 = (f2 - f0) * (f2 - f1)
    if abs(denom0) < 1e-30 or abs(denom1) < 1e-30 or abs(denom2) < 1e-30:
        return None
    x = (x0 * f1 * f2 / denom0 + x1 * f0 * f2 / denom1 + x2 * f0 * f1 / denom2)
    if not math.isfinite(x):
        return None
    # Check if x is in bracket
    if a is not None and b is not None:
        if not (a <= x <= b or b <= x <= a):
            return None
    return x
# EVOLVE-BLOCK-END
