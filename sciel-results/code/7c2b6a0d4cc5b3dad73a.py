"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def find_root(f, x0):
    """Find x such that |f(x)| <= 1e-8 using bracketing + regula falsi with bisection fallback."""
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    tol = 1e-10  # tighter than required 1e-8 for safety
    max_iter = 200

    x = float(x0)
    fx = f(x)
    if abs(fx) <= 1e-8:
        return x

    # Try to find a bracket [a, b] with sign change
    a, b = x, x
    fa, fb = fx, fx
    found_bracket = False

    # Expand bracket adaptively
    step = 1.0
    for _ in range(60):
        if time.time() > deadline:
            break
        # Try expanding to the right
        x_right = x + step
        fx_right = f(x_right)
        if abs(fx_right) <= 1e-8:
            return x_right
        if fa * fx_right < 0:
            a, b = x, x_right
            fa, fb = fx, fx_right
            found_bracket = True
            break
        # Try expanding to the left
        x_left = x - step
        fx_left = f(x_left)
        if abs(fx_left) <= 1e-8:
            return x_left
        if fa * fx_left < 0:
            a, b = x_left, x
            fa, fb = fx_left, fx
            found_bracket = True
            break
        step *= 2.0
        if step > 1e15:
            break

    if not found_bracket:
        # Fallback: try a few more points and return best
        best_x, best_f = x, fx
        for dx in [-10, -1, -0.1, 0.1, 1, 10]:
            if time.time() > deadline:
                break
            xc = x + dx
            fc = f(xc)
            if abs(fc) < abs(best_f):
                best_x, best_f = xc, fc
        return best_x

    # Illinois/Regula Falsi with bisection fallback
    best_x, best_f = (a, fa) if abs(fa) < abs(fb) else (b, fb)

    for i in range(max_iter):
        if time.time() > deadline:
            break
        if abs(fa) <= tol:
            return a
        if abs(fb) <= tol:
            return b

        # Regula falsi step
        if abs(fa) < abs(fb):
            # a is closer to root; shrink b side
            x_new = b - fb * (b - a) / (fb - fa)
            x_new = max(a + 1e-15 * abs(b - a), min(x_new, b - 1e-15 * abs(b - a)))
            fx_new = f(x_new)
            if abs(fx_new) <= tol:
                return x_new
            if fa * fx_new < 0:
                b = x_new
                fb = fx_new
            else:
                a = x_new
                fa = fx_new
                # Illinois: halve the side that was just rejected
                fa *= 0.5
        else:
            # b is closer to root; shrink a side
            x_new = a - fa * (b - a) / (fb - fa)
            x_new = max(a + 1e-15 * abs(b - a), min(x_new, b - 1e-15 * abs(b - a)))
            fx_new = f(x_new)
            if abs(fx_new) <= tol:
                return x_new
            if fa * fx_new < 0:
                b = x_new
                fb = fx_new
            else:
                a = x_new
                fa = fx_new
                # Illinois: halve the side that was just rejected
                fb *= 0.5

        # Track best
        if abs(fa) < abs(best_f):
            best_x, best_f = a, fa
        if abs(fb) < abs(best_f):
            best_x, best_f = b, fb

        # Bisection fallback if interval isn't shrinking
        if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
            mid = (a + b) / 2.0
            fmid = f(mid)
            if abs(fmid) <= tol:
                return mid
            if fa * fmid < 0:
                b = mid
                fb = fmid
            else:
                a = mid
                fa = fmid

    # Final check
    if abs(fa) <= 1e-8:
        return a
    if abs(fb) <= 1e-8:
        return b
    return best_x
# EVOLVE-BLOCK-END
