import math
import time

def find_root(f, x0):
    tol = 1e-8
    deadline = time.time() + 110.0  # Safety margin before 120s limit

    x = float(x0)
    fx = f(x)
    if abs(fx) <= tol:
        return x

    # Track best point found
    best_x = x
    best_f = abs(fx)

    # Try to find a bracket [a, b] where f(a) * f(b) < 0
    a = x
    fa = fx

    # First check nearby points for a sign change
    deltas = [1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

    bracket_found = False
    b = None
    fb = None

    for delta in deltas:
        if time.time() > deadline:
            return best_x

        # Check x + delta
        x_plus = x + delta
        f_plus = f(x_plus)
        if abs(f_plus) <= tol:
            return x_plus
        if abs(f_plus) < best_f:
            best_x = x_plus
            best_f = abs(f_plus)
        if fa * f_plus < 0:
            b = x_plus
            fb = f_plus
            bracket_found = True
            break

        # Check x - delta
        x_minus = x - delta
        f_minus = f(x_minus)
        if abs(f_minus) <= tol:
            return x_minus
        if abs(f_minus) < best_f:
            best_x = x_minus
            best_f = abs(f_minus)
        if fa * f_minus < 0:
            b = x_minus
            fb = f_minus
            bracket_found = True
            break

    # If no bracket found with initial deltas, try larger geometric expansion
    if not bracket_found:
        step = 1.0
        for i in range(50):
            if time.time() > deadline:
                return best_x

            for sign in [1, -1]:
                x_try = x + sign * step
                f_try = f(x_try)
                if abs(f_try) <= tol:
                    return x_try
                if abs(f_try) < best_f:
                    best_x = x_try
                    best_f = abs(f_try)
                if fa * f_try < 0:
                    b = x_try
                    fb = f_try
                    bracket_found = True
                    break
            if bracket_found:
                break
            step *= 2.0

    if not bracket_found:
        # No bracket found; return the point with smallest |f|
        return best_x

    # Ensure lo < hi
    lo = a
    hi = b
    flo = fa
    fhi = fb
    if lo > hi:
        lo, hi = hi, lo
        flo, fhi = fhi, flo

    # Hybrid Secant/Bisection: use bracket endpoints for secant formula
    for i in range(100):
        if time.time() > deadline:
            # Return midpoint of current bracket as best estimate
            mid = (lo + hi) / 2.0
            return mid

        if abs(hi - lo) < 1e-13:
            break

        # Attempt Secant Step using bracket endpoints
        denom = fhi - flo
        if abs(denom) > 1e-300:
            x_secant = hi - fhi * (hi - lo) / denom
            # Check if x_secant is strictly inside (lo, hi)
            if lo < x_secant < hi:
                x_new = x_secant
            else:
                x_new = (lo + hi) / 2.0
        else:
            x_new = (lo + hi) / 2.0

        f_new = f(x_new)

        if abs(f_new) <= tol:
            return x_new

        if abs(f_new) < best_f:
            best_x = x_new
            best_f = abs(f_new)

        # Update Bracket based on sign change
        if flo * f_new < 0:
            hi = x_new
            fhi = f_new
        else:
            lo = x_new
            flo = f_new

    # Final polish: do a few more bisection steps with tighter tolerance
    # to ensure we're well within 1e-8
    for i in range(20):
        if time.time() > deadline:
            break
        if abs(hi - lo) < 1e-15:
            break
        mid = (lo + hi) / 2.0
        fmid = f(mid)
        if abs(fmid) <= tol:
            return mid
        if abs(fmid) < best_f:
            best_x = mid
            best_f = abs(fmid)
        if flo * fmid < 0:
            hi = mid
            fhi = fmid
        else:
            lo = mid
            flo = fmid

    # Return the best point found
    return best_x
