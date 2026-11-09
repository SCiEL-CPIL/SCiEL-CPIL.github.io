"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    deadline = time.time() + 140  # 140s safety margin (evaluator kills at 150s)
    eval_count = 0
    max_evals = 500

    def safe_f(x):
        nonlocal eval_count
        if time.time() > deadline:
            raise TimeoutError()
        eval_count += 1
        if eval_count > max_evals:
            raise ValueError("max evals")
        val = f(x)
        if not math.isfinite(val):
            return val
        return float(val)

    x = float(x0)
    fx = safe_f(x)
    if abs(fx) <= 1e-8:
        return x

    # Find a bracket [a, b] where f(a) and f(b) have opposite signs
    a, b, fa, fb = None, None, None, None
    # Start with x as one endpoint, expand outward
    a, fa = x, fx
    # Try expanding in the direction of decreasing |f| first, then both directions
    h = 1e-4 * max(1.0, abs(x))
    
    # Try to find a bracket by expanding
    for attempt in range(100):
        if time.time() > deadline or eval_count >= max_evals:
            break
        # Try x + h and x - h
        for sign in (1, -1):
            if eval_count >= max_evals or time.time() > deadline:
                break
            candidate = x + sign * h
            fc = safe_f(candidate)
            if abs(fc) <= 1e-8:
                return candidate
            if math.isfinite(fc) and math.isfinite(fa) and fa * fc < 0:
                # Found bracket
                if sign == 1:
                    a, fa = x, fx
                    b, fb = candidate, fc
                else:
                    a, fa = candidate, fc
                    b, fb = x, fx
                break
            # If fc is smaller in magnitude, move in that direction
            if abs(fc) < abs(fa):
                a, fa = candidate, fc
                x = candidate
                fx = fc
        if a is not None and b is not None:
            break
        h *= 2.0
        if h > 1e6 * max(1.0, abs(x0)):
            break

    # If no bracket found, try a different strategy: use the best point found
    # and do a local search
    if a is None or b is None:
        # Fallback: try to bracket around x0 with various step sizes
        best_x, best_f = x, fx
        for h_try in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]:
            if time.time() > deadline or eval_count >= max_evals:
                break
            for sign in (1, -1):
                if eval_count >= max_evals or time.time() > deadline:
                    break
                cand = x0 + sign * h_try
                try:
                    fc = safe_f(cand)
                except (TimeoutError, ValueError):
                    break
                if abs(fc) <= 1e-8:
                    return cand
                if math.isfinite(fc) and math.isfinite(fx) and fx * fc < 0:
                    if sign == 1:
                        a, fa, b, fb = x0, fx, cand, fc
                    else:
                        a, fa, b, fb = cand, fc, x0, fx
                    break
                if math.isfinite(fc) and abs(fc) < abs(best_f):
                    best_x, best_f = cand, fc
            if a is not None:
                break

    if a is None or b is None:
        # No bracket found; return best point
        return best_x if 'best_x' in dir() else x

    # Ensure a < b
    if a > b:
        a, b = b, a
        fa, fb = fb, fa

    # Now use inverse quadratic interpolation with bisection fallback
    # We need at least 3 points for IQI; start with 2 and use secant/bisection
    # Keep track of the three most recent distinct points
    points = [(a, fa), (b, fb)]
    c = None
    fc = None

    for iteration in range(200):
        if time.time() > deadline or eval_count >= max_evals:
            break
        if abs(fa) <= 1e-8:
            return a
        if abs(fb) <= 1e-8:
            return b

        # Determine which endpoint has smaller |f|
        if abs(fa) <= abs(fb):
            # a is the "active" endpoint
            pass
        else:
            # b is the "active" endpoint
            pass

        # Try inverse quadratic interpolation if we have 3 points
        used_iqi = False
        if len(points) >= 3:
            # Use the three most recent points
            (x1, y1), (x2, y2), (x3, y3) = points[-3], points[-2], points[-1]
            # Check they are distinct
            if x1 != x2 and x2 != x3 and x1 != x3:
                try:
                    # Inverse quadratic interpolation: fit x as a function of y
                    # x = A*y^2 + B*y + C
                    # Using Lagrange interpolation
                    denom1 = (y1 - y2) * (y1 - y3)
                    denom2 = (y2 - y1) * (y2 - y3)
                    denom3 = (y3 - y1) * (y3 - y2)
                    if abs(denom1) > 1e-30 and abs(denom2) > 1e-30 and abs(denom3) > 1e-30:
                        x_new = (x1 * y2 * y3 / denom1 + x2 * y1 * y3 / denom2 + x3 * y1 * y2 / denom3)
                        # Accept if x_new is within the bracket and not too close to endpoints
                        if a < x_new < b:
                            # Also require it's not too close to existing points
                            min_dist = min(abs(x_new - p[0]) for p in points[-3:])
                            if min_dist > 1e-12 * max(1.0, abs(x_new)):
                                fc_new = safe_f(x_new)
                                if abs(fc_new) <= 1e-8:
                                    return x_new
                                if math.isfinite(fc_new):
                                    # Update bracket
                                    if fa * fc_new < 0:
                                        b, fb = x_new, fc_new
                                    elif fb * fc_new < 0:
                                        a, fa = x_new, fc_new
                                    points.append((x_new, fc_new))
                                    if len(points) > 5:
                                        points = points[-5:]
                                    used_iqi = True
                except (ZeroDivisionError, OverflowError, ValueError):
                    pass

        if used_iqi:
            continue

        # Fallback: bisection
        mid = (a + b) / 2.0
        if mid <= a or mid >= b:
            # Bracket is too small
            if abs(fa) <= abs(fb):
                return a
            else:
                return b
        fmid = safe_f(mid)
        if abs(fmid) <= 1e-8:
            return mid
        if not math.isfinite(fmid):
            # Try a different point
            mid = (a + b) / 2.0 + (b - a) * 1e-6
            fmid = safe_f(mid)
            if abs(fmid) <= 1e-8:
                return mid
            if not math.isfinite(fmid):
                break

        if fa * fmid < 0:
            b, fb = mid, fmid
        else:
            a, fa = mid, fmid

        points.append((mid, fmid))
        if len(points) > 5:
            points = points[-5:]

    # Return the best point
    if abs(fa) <= abs(fb):
        return a
    else:
        return b
# EVOLVE-BLOCK-END
