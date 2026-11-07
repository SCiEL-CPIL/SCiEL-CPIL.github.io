"""Root finder for smooth polynomial equations - secant-based hybrid."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit

    x = float(x0)

    # Evaluate initial point
    try:
        fx = f(x)
    except Exception:
        return x0

    if not math.isfinite(fx):
        # Try nearby points to find a finite value
        for dx in [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]:
            for sign in [1, -1]:
                xt = x + sign * dx
                try:
                    fxt = f(xt)
                except Exception:
                    continue
                if math.isfinite(fxt):
                    x = xt
                    fx = fxt
                    break
            if math.isfinite(fx):
                break
        if not math.isfinite(fx):
            return x0

    if abs(fx) <= 1e-8:
        return x

    # Secant method: need two points
    # Choose second point with adaptive step
    best_x = x
    best_fx = fx
    best_abs = abs(fx)

    # Try to find a good second point, preferring sign changes (bracketing)
    h_base = max(1e-3, abs(x) * 1e-3)
    found_second = False
    x1 = None
    fx1 = None
    has_bracket = False

    for h in [h_base, h_base * 0.1, h_base * 10, h_base * 0.01, h_base * 100,
              1e-2, 0.1, 1.0, 10.0, 100.0]:
        for sign in [1, -1]:
            xt = x + sign * h
            try:
                fxt = f(xt)
            except Exception:
                continue
            if not math.isfinite(fxt):
                continue
            if abs(fxt) < best_abs:
                best_x = xt
                best_fx = fxt
                best_abs = abs(fxt)
            if abs(fxt) <= 1e-8:
                return xt
            if abs(fxt - fx) > 1e-12:
                # Prefer sign change (bracket) for robustness
                if fx * fxt < 0:
                    x1 = xt
                    fx1 = fxt
                    found_second = True
                    has_bracket = True
                    break
                elif not found_second:
                    x1 = xt
                    fx1 = fxt
                    found_second = True
        if has_bracket:
            break

    if not found_second:
        return best_x

    # Now we have (x, fx) and (x1, fx1) with fx != fx1
    x0_s = x
    f0_s = fx
    x1_s = x1
    f1_s = fx1

    # Track bracket for bisection safeguard
    a, b = None, None
    if has_bracket:
        if fx * f1_s < 0:
            a, b = (x0_s, x1_s) if abs(f0_s) < abs(f1_s) else (x1_s, x0_s)

    # Hybrid secant/bisection iterations
    max_iter = 80
    for i in range(max_iter):
        if time.time() > deadline:
            break

        # Check convergence
        if abs(f1_s) <= 1e-8:
            return x1_s

        # Secant step: x_new = x1 - f1 * (x1 - x0) / (f1 - f0)
        denom = f1_s - f0_s
        x_new = None

        if abs(denom) >= 1e-14:
            step = f1_s * (x1_s - x0_s) / denom
            candidate = x1_s - step

            # Safeguard: don't go too far
            max_step = max(1.0, abs(x1_s))
            if abs(step) > max_step:
                candidate = x1_s + math.copysign(max_step, -step)

            # If we have a bracket, prefer staying within it
            if a is not None and b is not None:
                lo, hi = min(a, b), max(a, b)
                if lo < candidate < hi:
                    x_new = candidate
                else:
                    # Use bisection instead
                    x_new = 0.5 * (a + b)
            else:
                x_new = candidate
        else:
            # Nearly parallel: use bisection if bracket exists, else perturb
            if a is not None and b is not None:
                x_new = 0.5 * (a + b)
            else:
                x_new = x1_s + 1e-6

        try:
            f_new = f(x_new)
        except Exception:
            break

        if not math.isfinite(f_new):
            # Shrink or try bisection
            if a is not None and b is not None:
                x_new = 0.5 * (a + b)
                try:
                    f_new = f(x_new)
                except Exception:
                    break
                if not math.isfinite(f_new):
                    break
            else:
                break

        if abs(f_new) < best_abs:
            best_x = x_new
            best_fx = f_new
            best_abs = abs(f_new)

        if abs(f_new) <= 1e-8:
            return x_new

        # Update bracket if sign change
        if a is not None and b is not None:
            if f_new * f(a) <= 0:
                b = x_new
            elif f_new * f(b) <= 0:
                a = x_new
            # Recompute bracket endpoints' f values if needed
            # (we only track a, b as x values; use stored f when possible)

        # Update points: keep the two most recent
        x0_s = x1_s
        f0_s = f1_s
        x1_s = x_new
        f1_s = f_new

    # Final polish: if we have a good candidate, do a few Newton-like steps
    # using secant derivative estimate
    if abs(best_fx) <= 1e-6 and abs(best_fx) > 1e-8:
        xp = best_x
        fp = best_fx
        for _ in range(10):
            if time.time() > deadline:
                break
            # Estimate derivative via secant
            h = max(1e-6, abs(xp) * 1e-6)
            try:
                fp2 = f(xp + h)
            except Exception:
                break
            if not math.isfinite(fp2):
                break
            deriv = (fp2 - fp) / h
            if abs(deriv) < 1e-14:
                break
            x_new = xp - fp / deriv
            try:
                f_new = f(x_new)
            except Exception:
                break
            if not math.isfinite(f_new):
                break
            if abs(f_new) < best_abs:
                best_x = x_new
                best_fx = f_new
                best_abs = abs(f_new)
            if abs(f_new) <= 1e-8:
                return x_new
            xp = x_new
            fp = f_new

    return best_x
# EVOLVE-BLOCK-END
