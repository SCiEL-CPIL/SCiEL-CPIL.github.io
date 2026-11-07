"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    x = float(x0)

    # 1. Evaluate initial point
    try:
        fx = f(x)
    except Exception:
        return x0

    if not math.isfinite(fx):
        # Try to find a nearby finite point
        found = False
        for dx in [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]:
            for sign in [1, -1]:
                x_test = x + sign * dx
                try:
                    fx_test = f(x_test)
                except Exception:
                    continue
                if math.isfinite(fx_test):
                    x = x_test
                    fx = fx_test
                    found = True
                    break
            if found:
                break
        if not found:
            return x0

    if abs(fx) <= 1e-8:
        return x

    # Best point tracking
    best_x = x
    best_fx = fx
    best_abs_fx = abs(fx)

    # 2. Establish second point with geometric step
    h = max(1e-5, abs(x) * 1e-5)
    if h < 1e-8:
        h = 1e-5

    x2 = x + h
    try:
        f2 = f(x2)
    except Exception:
        x2 = x - h
        try:
            f2 = f(x2)
        except Exception:
            return best_x

    if not math.isfinite(f2):
        # Try more offsets
        found = False
        for dx in [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]:
            for sign in [1, -1]:
                x_test = x + sign * dx
                try:
                    fx_test = f(x_test)
                except Exception:
                    continue
                if math.isfinite(fx_test):
                    x2 = x_test
                    f2 = fx_test
                    found = True
                    break
            if found:
                break
        if not found:
            return best_x

    if abs(f2) <= 1e-8:
        return x2
    if abs(f2) < best_abs_fx:
        best_x = x2
        best_fx = f2
        best_abs_fx = abs(f2)

    # History for quadratic interpolation: list of (x, f)
    # We need at least 3 points for quadratic interpolation
    history = [(x, fx), (x2, f2)]

    # 3. Main iteration: alternate between secant and quadratic interpolation
    max_iter = 50

    for i in range(max_iter):
        if time.time() > deadline:
            break

        # Check if we have enough points for quadratic interpolation
        if len(history) >= 3:
            x_a, f_a = history[-3]
            x_b, f_b = history[-2]
            x_c, f_c = history[-1]

            # Check for degenerate cases (duplicate x values)
            if abs(x_a - x_b) > 1e-15 and abs(x_b - x_c) > 1e-15 and abs(x_a - x_c) > 1e-15:
                # Quadratic interpolation: find root of parabola through (x_a,f_a), (x_b,f_b), (x_c,f_c)
                # Using Lagrange form or solving the quadratic
                # q(t) = f_a * (t-x_b)(t-x_c)/((x_a-x_b)(x_a-x_c))
                #       + f_b * (t-x_a)(t-x_c)/((x_b-x_a)(x_b-x_c))
                #       + f_c * (t-x_a)(t-x_b)/((x_c-x_a)(x_c-x_b))
                # Expand: q(t) = At^2 + Bt + C
                # A = f_a/((x_a-x_b)(x_a-x_c)) + f_b/((x_b-x_a)(x_b-x_c)) + f_c/((x_c-x_a)(x_c-x_b))
                # B = -f_a*(x_b+x_c)/((x_a-x_b)(x_a-x_c)) - f_b*(x_a+x_c)/((x_b-x_a)(x_b-x_c)) - f_c*(x_a+x_b)/((x_c-x_a)(x_c-x_b))
                # C = f_a*x_b*x_c/((x_a-x_b)(x_a-x_c)) + f_b*x_a*x_c/((x_b-x_a)(x_b-x_c)) + f_c*x_a*x_b/((x_c-x_a)(x_c-x_b))

                da = (x_a - x_b) * (x_a - x_c)
                db = (x_b - x_a) * (x_b - x_c)
                dc = (x_c - x_a) * (x_c - x_b)

                if abs(da) > 1e-30 and abs(db) > 1e-30 and abs(dc) > 1e-30:
                    A = f_a / da + f_b / db + f_c / dc
                    B = (-f_a * (x_b + x_c)) / da + (-f_b * (x_a + x_c)) / db + (-f_c * (x_a + x_b)) / dc
                    C = (f_a * x_b * x_c) / da + (f_b * x_a * x_c) / db + (f_c * x_a * x_b) / dc

                    # Solve At^2 + Bt + C = 0
                    if abs(A) < 1e-30:
                        # Linear case
                        if abs(B) > 1e-30:
                            x_new = -C / B
                        else:
                            # Degenerate, fall back to secant
                            x_new = None
                    else:
                        disc = B * B - 4.0 * A * C
                        if disc < 0:
                            # No real root, fall back to secant
                            x_new = None
                        else:
                            sqrt_disc = math.sqrt(disc)
                            # Choose the root that is closer to x_c (most recent point)
                            r1 = (-B + sqrt_disc) / (2.0 * A)
                            r2 = (-B - sqrt_disc) / (2.0 * A)
                            # Pick the root closer to the most recent x
                            if abs(r1 - x_c) < abs(r2 - x_c):
                                x_new = r1
                            else:
                                x_new = r2
                    if x_new is not None and math.isfinite(x_new):
                        # Safeguard: limit step size
                        max_step = max(1.0, abs(x_c) * 0.5)
                        if abs(x_new - x_c) > max_step:
                            x_new = x_c + math.copysign(max_step, x_new - x_c)

                        try:
                            f_new = f(x_new)
                        except Exception:
                            f_new = None

                        if f_new is not None and math.isfinite(f_new):
                            if abs(f_new) < best_abs_fx:
                                best_x = x_new
                                best_fx = f_new
                                best_abs_fx = abs(f_new)
                            if abs(f_new) <= 1e-8:
                                return x_new

                            # Add to history
                            history.append((x_new, f_new))
                            if len(history) > 5:
                                history = history[-5:]
                            continue
                        else:
                            # Non-finite, fall back to secant below
                            pass
                    # If quadratic failed, fall through to secant

        # Secant method (fallback or when < 3 points)
        if len(history) >= 2:
            x_prev, f_prev = history[-2]
            x_curr, f_curr = history[-1]
        else:
            # Should not happen, but safeguard
            break

        denom = f_curr - f_prev
        if abs(denom) < 1e-15:
            # Perturb to escape
            x_curr = x_curr + 1e-4
            try:
                f_curr = f(x_curr)
            except Exception:
                break
            if not math.isfinite(f_curr):
                break
            if abs(f_curr) < best_abs_fx:
                best_x = x_curr
                best_fx = f_curr
                best_abs_fx = abs(f_curr)
            if abs(f_curr) <= 1e-8:
                return x_curr
            history.append((x_curr, f_curr))
            if len(history) > 5:
                history = history[-5:]
            continue

        # Secant step
        x_new = x_curr - f_curr * (x_curr - x_prev) / denom

        # Safeguard: limit step size
        max_step = max(1.0, abs(x_curr) * 0.3)
        if abs(x_new - x_curr) > max_step:
            x_new = x_curr + math.copysign(max_step, x_new - x_curr)

        try:
            f_new = f(x_new)
        except Exception:
            break

        if not math.isfinite(f_new):
            # Bisection fallback
            x_new = 0.5 * (x_curr + x_prev)
            try:
                f_new = f(x_new)
            except Exception:
                break
            if not math.isfinite(f_new):
                break

        if abs(f_new) < best_abs_fx:
            best_x = x_new
            best_fx = f_new
            best_abs_fx = abs(f_new)

        if abs(f_new) <= 1e-8:
            return x_new

        # Stagnation check
        if abs(x_new - x_curr) < 1e-13 * max(1.0, abs(x_curr)):
            # Small perturbation
            x_new = x_curr + 1e-5
            try:
                f_new = f(x_new)
            except Exception:
                break
            if not math.isfinite(f_new):
                break
            if abs(f_new) < best_abs_fx:
                best_x = x_new
                best_fx = f_new
                best_abs_fx = abs(f_new)
            if abs(f_new) <= 1e-8:
                return x_new

        history.append((x_new, f_new))
        if len(history) > 5:
            history = history[-5:]

    return best_x
# EVOLVE-BLOCK-END
