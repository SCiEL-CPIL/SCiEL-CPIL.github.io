"""Scalar root finder on 3 fixed oscillatory equations.

Strategy:
  1. Evaluate f at x0. If |f| <= 1e-8, return immediately.
  2. Find a sign-change bracket via multi-start local search.
  3. Use Illinois-style inverse linear interpolation (damped bisection)
     to converge to |f| <= 1e-10.
  4. If no bracket found, use damped Newton local minimization of |f(x)|.
  5. Track the best (smallest |f|) point and return it on deadline.
"""
import time
import math

# EVOLVE-BLOCK-START
_START_TIME = None
_DEADLINE = 145.0

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _is_finite(x):
    return math.isfinite(x)

def find_root(f, x0):
    """
    Find a root x of f such that |f(x)| <= 1e-8.
    Returns a float x.
    """
    x0 = float(x0)
    if not _is_finite(x0):
        x0 = 0.0

    best_x = x0
    best_f = None
    try:
        best_f = f(x0)
    except:
        best_f = float('inf')

    if best_f is not None and _is_finite(best_f) and abs(best_f) <= 1e-8:
        return x0

    # Phase 1: Find a sign-change bracket
    a = x0
    fa = best_f
    if fa is None or not _is_finite(fa):
        for dx in [1.0, -1.0, 0.5, -0.5, 2.0, -2.0]:
            try:
                fx_try = f(x0 + dx)
                if _is_finite(fx_try):
                    a = x0 + dx
                    fa = fx_try
                    break
            except:
                continue

    if fa is None or not _is_finite(fa):
        return x0

    if abs(fa) <= 1e-8:
        return a

    b = None
    fb = None

    # Try nearby points with strategic spacing
    for dx in [0.1, -0.1, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 5.0, -5.0, 10.0, -10.0]:
        try:
            if _elapsed() > _DEADLINE:
                break
            b_try = a + dx
            fb_try = f(b_try)
            if _is_finite(fb_try):
                if abs(fb_try) <= 1e-8:
                    return b_try
                if fa * fb_try < 0:
                    b = b_try
                    fb = fb_try
                    break
        except:
            continue

    # If no bracket found, try expanding geometrically
    if b is None:
        for scale in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
            for sign in [1, -1]:
                try:
                    if _elapsed() > _DEADLINE:
                        break
                    b_try = x0 + sign * scale * max(1.0, abs(x0))
                    fb_try = f(b_try)
                    if _is_finite(fb_try):
                        if abs(fb_try) <= 1e-8:
                            return b_try
                        if _is_finite(fa) and fa * fb_try < 0:
                            b = b_try
                            fb = fb_try
                            break
                except:
                    continue
            if b is not None:
                break

    # Phase 2: Illinois-style interpolation if bracket found
    if b is not None and fb is not None and fa * fb < 0:
        left = a
        fleft = fa
        right = b
        fright = fb

        # Illinois method: inverse linear interpolation with damping
        # When the new point is on the same side as the previous rejected point,
        # halve the distance instead of using full interpolation.
        x_prev = right
        f_prev = fright

        for _ in range(100):
            if _elapsed() > _DEADLINE:
                break

            if abs(fleft) <= 1e-10:
                return left
            if abs(fright) <= 1e-10:
                return right

            # Inverse linear interpolation
            if f_prev != fleft and (right - x_prev) != 0:
                x_interp = right - fright * (right - x_prev) / (fright - f_prev)
            else:
                x_interp = (left + right) / 2.0

            # Check if interpolation point is within bracket
            if left < x_interp < right:
                try:
                    f_interp = f(x_interp)
                except:
                    f_interp = None

                if f_interp is not None and _is_finite(f_interp):
                    if abs(f_interp) <= 1e-8:
                        return x_interp

                    if f_interp * fleft < 0:
                        # New point is in (left, x_interp)
                        right = x_interp
                        fright = f_interp
                        x_prev = right
                        f_prev = fright
                    elif f_interp * fright < 0:
                        # New point is in (x_interp, right)
                        left = x_interp
                        fleft = f_interp
                        x_prev = left
                        f_prev = fleft
                    else:
                        # Interpolation failed (same sign), use bisection
                        mid = (left + right) / 2.0
                        try:
                            f_mid = f(mid)
                        except:
                            f_mid = None
                        if f_mid is not None and _is_finite(f_mid):
                            if abs(f_mid) <= 1e-8:
                                return mid
                            if f_mid * fleft < 0:
                                right = mid
                                fright = f_mid
                                x_prev = right
                                f_prev = fright
                            else:
                                left = mid
                                fleft = f_mid
                                x_prev = left
                                f_prev = fleft
                        else:
                            # Can't evaluate, break
                            break
                else:
                    # Can't evaluate, use bisection
                    mid = (left + right) / 2.0
                    try:
                        f_mid = f(mid)
                    except:
                        f_mid = None
                    if f_mid is not None and _is_finite(f_mid):
                        if abs(f_mid) <= 1e-8:
                            return mid
                        if f_mid * fleft < 0:
                            right = mid
                            fright = f_mid
                            x_prev = right
                            f_prev = fright
                        else:
                            left = mid
                            fleft = f_mid
                            x_prev = left
                            f_prev = fleft
                    else:
                        break
            else:
                # Interpolation went outside bracket, use bisection
                mid = (left + right) / 2.0
                try:
                    f_mid = f(mid)
                except:
                    f_mid = None
                if f_mid is not None and _is_finite(f_mid):
                    if abs(f_mid) <= 1e-8:
                        return mid
                    if f_mid * fleft < 0:
                        right = mid
                        fright = f_mid
                        x_prev = right
                        f_prev = fright
                    else:
                        left = mid
                        fleft = f_mid
                        x_prev = left
                        f_prev = fleft
                else:
                    break

            # Convergence check
            if abs(right - left) < 1e-14 * max(1.0, abs(left), abs(right)):
                break

            # Check if we've converged in function value
            if abs(fright) <= 1e-8:
                return right
            if abs(fleft) <= 1e-8:
                return left

        # Return the point with smallest |f|
        if abs(fleft) <= abs(fright):
            return left
        else:
            return right

    # Phase 3: No bracket found, use damped Newton local search
    x = x0
    fx = best_f
    if fx is None or not _is_finite(fx):
        for dx in [0.1, -0.1, 0.5, -0.5, 1.0, -1.0]:
            try:
                x_try = x0 + dx
                fx_try = f(x_try)
                if _is_finite(fx_try):
                    x = x_try
                    fx = fx_try
                    break
            except:
                continue

    if fx is None or not _is_finite(fx):
        return x0

    for _ in range(80):
        if _elapsed() > _DEADLINE:
            break

        if abs(fx) <= 1e-8:
            return x

        # Finite difference derivative
        h = 1e-7 * max(1.0, abs(x))
        try:
            fx_plus = f(x + h)
        except:
            fx_plus = None

        if fx_plus is not None and _is_finite(fx_plus):
            deriv = (fx_plus - fx) / h
            if deriv != 0 and _is_finite(deriv):
                x_new = x - fx / deriv
                if _is_finite(x_new):
                    # Damped step
                    if abs(x_new - x) > max(1.0, abs(x)):
                        x_new = x + 0.5 * (x_new - x) * (1 if x_new > x else -1)

                    try:
                        fx_new = f(x_new)
                    except:
                        fx_new = None

                    if fx_new is not None and _is_finite(fx_new):
                        if abs(fx_new) <= 1e-8:
                            return x_new
                        if abs(fx_new) < abs(fx):
                            x = x_new
                            fx = fx_new
                        else:
                            x_new = x - 0.1 * fx / deriv
                            if _is_finite(x_new):
                                try:
                                    fx_new = f(x_new)
                                except:
                                    fx_new = None
                                if fx_new is not None and _is_finite(fx_new):
                                    if abs(fx_new) <= 1e-8:
                                        return x_new
                                    if abs(fx_new) < abs(fx):
                                        x = x_new
                                        fx = fx_new
        else:
            break

    # Return best found
    if fx is not None and _is_finite(fx) and abs(fx) <= 1e-8:
        return x

    return x

# EVOLVE-BLOCK-END
