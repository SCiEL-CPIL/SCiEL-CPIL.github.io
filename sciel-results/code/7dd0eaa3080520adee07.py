"""
Scalar root finder on 3 fixed oscillatory equations.

Strategy:
  1. Evaluate f at x0. If |f| <= 1e-8, return immediately.
  2. Use a walking bracket finder with midpoint checks to find a sign-change
     bracket robustly, even for fast oscillations.
  3. Use Brent's method (inverse quadratic + secant + bisection) to converge.
  4. If no bracket is found, fall back to secant/Newton iteration with
     safeguarded steps.
  5. Track the best point and return it if the deadline is approached.

All f calls are counted by the evaluator. The final check is not counted.
"""
import time
import math

# EVOLVE-BLOCK-START
_START_TIME = None
_DEADLINE = 145.0  # 145s out of 150s wall-clock limit

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _is_finite(x):
    return math.isfinite(x)

def _brent_root(f, a, fa, b, fb, max_iter=80):
    """
    Brent's method to find a root in [a, b] with fa*fb < 0.
    Returns x such that |f(x)| <= 1e-10 or best effort.
    """
    # Ensure a < b
    if a > b:
        a, b = b, a
        fa, fb = fb, fa

    c = a
    fc = fa
    d = b - a
    e = d

    for _ in range(max_iter):
        if _elapsed() > _DEADLINE:
            break

        if abs(fb) <= 1e-10:
            return b
        if abs(fc) <= 1e-10:
            return c

        # Ensure c is the point with smaller |f|
        if abs(fc) > abs(fb):
            a, b = b, a
            fa, fb = fb, fa
            c = a
            fc = fa
            d = b - a
            e = d

        # Set up tolerance
        tol1 = 2 * 2.22e-16 * max(abs(a), 1.0)
        half_e = e / 2.0

        if abs(c - a) <= tol1 or fc == 0:
            return a

        # Inverse quadratic interpolation
        do_interp = True
        if a == c:
            S = fb / fa
            m = S / 2.0 * (c - a)
            if not (c - abs(m) < a and a < c + abs(m)):
                m = half_e * (1.0 if m >= 0 else -1.0)
            do_interp = False
        elif b == c:
            do_interp = False
        else:
            S = fb / fa
            P = S * (c - b)
            Q = 1.0 - S
            # Compute the interpolation point
            # x = b + (0.5 * (c - b) * Q / (Q - S) + P)
            # But we need to be careful
            m_val = (0.5 * (c - b) * Q) / (Q - S) + P
            # Check conditions
            if not (b - abs(m_val) < a and a < b + abs(m_val)):
                m_val = half_e * (1.0 if m_val >= 0 else -1.0)
                do_interp = False
            else:
                if abs(m_val) >= abs(half_e):
                    m_val = half_e * (1.0 if m_val >= 0 else -1.0)
                    do_interp = False

        if do_interp:
            x_new = b + m_val
        else:
            x_new = b + half_e

        e = d
        d = b - a

        try:
            fx = f(x_new)
        except:
            fx = float('inf')

        if not _is_finite(fx):
            # Fallback to bisection
            x_new = (a + b) / 2.0
            try:
                fx = f(x_new)
            except:
                break
            if not _is_finite(fx):
                break

        if abs(fx) <= 1e-10:
            return x_new

        c = b
        fc = fb
        b = x_new
        fb = fx

        if fa * fb < 0:
            a = c
            fa = fc

    # Return the best estimate
    if abs(fa) <= abs(fb):
        return a
    elif abs(fc) <= abs(fb):
        return c
    else:
        return b

def find_root(f, x0):
    """
    Find a root x of f such that |f(x)| <= 1e-8.
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

    # Try to find a valid starting point if x0 is invalid
    a = x0
    fa = best_f
    if fa is None or not _is_finite(fa):
        found_valid = False
        for dx in [0.1, -0.1, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0]:
            try:
                fx_try = f(x0 + dx)
                if _is_finite(fx_try):
                    a = x0 + dx
                    fa = fx_try
                    found_valid = True
                    break
            except:
                continue
        if not found_valid:
            return x0

    if abs(fa) <= 1e-8:
        return a

    # Walking bracket finder
    # Try multiple step sizes, walking in both directions
    # For each step, check midpoint to avoid jumping over roots
    step_sizes = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]

    for direction in [1, -1]:
        if _elapsed() > _DEADLINE:
            break

        prev_x = a
        prev_f = fa

        for s in step_sizes:
            if _elapsed() > _DEADLINE:
                break

            target = x0 + direction * s

            # Check midpoint to avoid jumping over roots
            mid_x = (prev_x + target) / 2.0
            try:
                f_mid = f(mid_x)
            except:
                f_mid = None

            if f_mid is not None and _is_finite(f_mid):
                if abs(f_mid) <= 1e-8:
                    return mid_x
                if prev_f * f_mid < 0:
                    # Bracket [prev_x, mid_x]
                    try:
                        return _brent_root(f, prev_x, prev_f, mid_x, f_mid)
                    except:
                        # Bisection fallback
                        left, f_left = prev_x, prev_f
                        right, f_right = mid_x, f_mid
                        if left > right:
                            left, right = right, left
                            f_left, f_right = f_right, f_left
                        for _ in range(60):
                            if _elapsed() > _DEADLINE:
                                break
                            mid2 = (left + right) / 2.0
                            try:
                                f_mid2 = f(mid2)
                            except:
                                break
                            if not _is_finite(f_mid2):
                                break
                            if abs(f_mid2) <= 1e-8:
                                return mid2
                            if f_left * f_mid2 < 0:
                                right, f_right = mid2, f_mid2
                            else:
                                left, f_left = mid2, f_mid2
                        return left if abs(f_left) < abs(f_right) else right

            # Check target
            try:
                f_target = f(target)
            except:
                f_target = None

            if f_target is None or not _is_finite(f_target):
                # Update prev to mid if we have a valid one
                if f_mid is not None and _is_finite(f_mid):
                    prev_x = mid_x
                    prev_f = f_mid
                continue

            if abs(f_target) <= 1e-8:
                return target

            if prev_f * f_target < 0:
                # Bracket [prev_x, target]
                try:
                    return _brent_root(f, prev_x, prev_f, target, f_target)
                except:
                    # Bisection fallback
                    left, f_left = prev_x, prev_f
                    right, f_right = target, f_target
                    if left > right:
                        left, right = right, left
                        f_left, f_right = f_right, f_left
                    for _ in range(60):
                        if _elapsed() > _DEADLINE:
                            break
                        mid2 = (left + right) / 2.0
                        try:
                            f_mid2 = f(mid2)
                        except:
                            break
                        if not _is_finite(f_mid2):
                            break
                        if abs(f_mid2) <= 1e-8:
                            return mid2
                        if f_left * f_mid2 < 0:
                            right, f_right = mid2, f_mid2
                        else:
                            left, f_left = mid2, f_mid2
                    return left if abs(f_left) < abs(f_right) else right

            # Update prev
            prev_x = target
            prev_f = f_target

    # If no bracket found, try secant method with safeguarded steps
    # This is a fallback for cases where the function doesn't have a sign change
    # but has a root (e.g., tangent)
    if _elapsed() < _DEADLINE:
        # Try secant from a few starting points
        for start_offset in [0.1, -0.1, 0.5, -0.5, 1.0, -1.0]:
            if _elapsed() > _DEADLINE:
                break

            x1 = x0 + start_offset
            try:
                f1 = f(x1)
            except:
                continue

            if not _is_finite(f1):
                continue

            if abs(f1) <= 1e-8:
                return x1

            x2 = x0 + 2.0 * start_offset
            try:
                f2 = f(x2)
            except:
                continue

            if not _is_finite(f2):
                continue

            if abs(f2) <= 1e-8:
                return x2

            # Secant iteration
            x_prev = x1
            f_prev = f1
            x_curr = x2
            f_curr = f2

            for _ in range(40):
                if _elapsed() > _DEADLINE:
                    break

                if abs(f_curr) <= 1e-8:
                    return x_curr

                if f_curr == f_prev:
                    break

                # Secant step
                x_next = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)

                # Safeguard: limit step size
                max_step = 2.0 * abs(x_curr) + 10.0
                if abs(x_next - x_curr) > max_step:
                    x_next = x_curr + max_step * (1.0 if x_next > x_curr else -1.0)

                try:
                    f_next = f(x_next)
                except:
                    break

                if not _is_finite(f_next):
                    break

                if abs(f_next) <= 1e-8:
                    return x_next

                x_prev = x_curr
                f_prev = f_curr
                x_curr = x_next
                f_curr = f_next

                # If the step is too small, break
                if abs(x_next - x_curr) < 1e-14 * max(1.0, abs(x_curr)):
                    break

    return x0

# EVOLVE-BLOCK-END
