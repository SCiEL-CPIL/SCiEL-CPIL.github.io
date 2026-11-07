"""Refined root finder: improved initial point selection, adaptive stall handling, and final polish."""
import time
import math

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin under 150s

    x = float(x0)
    call_count = 0

    def remaining():
        return deadline - time.time()

    # Get first evaluation
    try:
        fx = f(x)
        call_count += 1
    except Exception:
        return x

    if not math.isfinite(fx):
        return x
    if abs(fx) <= 1e-8:
        return x

    best_x = x
    best_fx = fx

    # Try to find a good second point for secant
    # We want a point where the secant slope is meaningful and the step won't be too large
    step_factors = [1e-4, 1e-3, 1e-2, 0.1, 1.0, -1e-4, -1e-3, -1e-2, -0.1, -1.0,
                    2.0, -2.0, 5.0, -5.0, 0.01, -0.01]
    x2 = None
    fx2 = None
    best_secant_score = None

    for sf in step_factors:
        if remaining() < 1.0:
            return best_x
        h = sf * max(1.0, abs(x))
        x_try = x + h
        try:
            fx_try = f(x_try)
            call_count += 1
        except Exception:
            continue
        if not math.isfinite(fx_try):
            continue
        if abs(fx_try) <= 1e-8:
            return x_try
        if abs(fx_try) < abs(best_fx):
            best_x = x_try
            best_fx = fx_try
        # Check if this gives a meaningful secant
        denom = fx_try - fx
        if abs(denom) > 1e-15 * max(1.0, abs(fx), abs(fx_try)):
            # Score: prefer smaller |h| (local approximation) and reasonable |denom|
            # The secant step would be: x_try - fx_try * (x_try - x) / denom
            # We want this step to not be too large relative to |x|
            secant_step = fx_try * h / denom
            # Prefer where secant_step is not too large and denom is not too small
            score = abs(secant_step) / max(1.0, abs(x_try)) + 1e-10 / max(abs(denom), 1e-30)
            if best_secant_score is None or score < best_secant_score:
                best_secant_score = score
                x2 = x_try
                fx2 = fx_try
            # If we have a very good local approximation, break early
            if abs(h) < 1e-3 * max(1.0, abs(x)) and abs(denom) > 1e-10:
                x2 = x_try
                fx2 = fx_try
                break

    if x2 is None or fx2 is None:
        # Couldn't find a good second point, try larger steps
        for scale in [10.0, 100.0, 1000.0, 10000.0]:
            if remaining() < 1.0:
                return best_x
            h = scale * max(1.0, abs(x))
            for sign in [1.0, -1.0]:
                x_try = x + sign * h
                try:
                    fx_try = f(x_try)
                    call_count += 1
                except Exception:
                    continue
                if not math.isfinite(fx_try):
                    continue
                if abs(fx_try) <= 1e-8:
                    return x_try
                if abs(fx_try) < abs(best_fx):
                    best_x = x_try
                    best_fx = fx_try
                if abs(fx_try - fx) > 1e-15 * max(1.0, abs(fx), abs(fx_try)):
                    x2 = x_try
                    fx2 = fx_try
                    break
            if x2 is not None:
                break

    if x2 is None or fx2 is None:
        return best_x

    # Secant method with adaptive stall detection
    prev_best = abs(best_fx)
    stall_count = 0
    perturbation_base = 0.1
    perturbation_power = 0

    max_iter = 495 - call_count
    if max_iter < 10:
        max_iter = 10

    for i in range(max_iter):
        if remaining() < 0.5:
            break

        denom = fx2 - fx
        if abs(denom) < 1e-30:
            # Nearly parallel; perturb
            h2 = perturbation_base * (2.0 ** perturbation_power) * max(1.0, abs(x2))
            x_new = x2 + h2
            try:
                fx_new = f(x_new)
                call_count += 1
            except Exception:
                return best_x
            if not math.isfinite(fx_new):
                return best_x
            if abs(fx_new) < abs(best_fx):
                best_x = x_new
                best_fx = fx_new
            if abs(fx_new) <= 1e-8:
                return x_new
            x, fx = x2, fx2
            x2, fx2 = x_new, fx_new
            stall_count = 0
            perturbation_power += 1
            continue

        x_new = x2 - fx2 * (x2 - x) / denom
        if not math.isfinite(x_new):
            return best_x

        try:
            fx_new = f(x_new)
            call_count += 1
        except Exception:
            return best_x

        if not math.isfinite(fx_new):
            return best_x

        improved = False
        if abs(fx_new) < abs(best_fx):
            best_x = x_new
            best_fx = fx_new
            improved = True
            stall_count = 0
        else:
            stall_count += 1

        if abs(fx_new) <= 1e-8:
            return x_new

        # If we've stalled, try an adaptive perturbation
        if stall_count >= 4:
            h3 = perturbation_base * (2.0 ** perturbation_power) * max(1.0, abs(best_x))
            # Alternate direction
            if (i % 2 == 1):
                h3 = -h3
            x_pert = best_x + h3
            try:
                fx_pert = f(x_pert)
                call_count += 1
            except Exception:
                return best_x
            if math.isfinite(fx_pert):
                if abs(fx_pert) < abs(best_fx):
                    best_x = x_pert
                    best_fx = fx_pert
                if abs(fx_pert) <= 1e-8:
                    return x_pert
                x, fx = x2, fx2
                x2, fx2 = x_pert, fx_pert
            perturbation_power += 1
            stall_count = 0
            continue

        # Shift: x = x2, x2 = x_new
        x, fx = x2, fx2
        x2, fx2 = x_new, fx_new

        # Reset perturbation power if we made good progress
        if improved and abs(fx_new) < abs(best_fx) * 0.5:
            perturbation_power = 0

    # Final polish: if we have budget, do a few more Newton-like steps
    if remaining() > 0.5 and call_count < 495:
        # Use the best point and try a local Newton step
        x_curr = best_x
        f_curr = best_fx
        # Estimate derivative via central difference
        h_deriv = 1e-6 * max(1.0, abs(x_curr))
        try:
            f_plus = f(x_curr + h_deriv)
            call_count += 1
            if math.isfinite(f_plus):
                f_minus = f(x_curr - h_deriv)
                call_count += 1
                if math.isfinite(f_minus):
                    deriv = (f_plus - f_minus) / (2.0 * h_deriv)
                    if abs(deriv) > 1e-30:
                        # Newton step
                        x_newton = x_curr - f_curr / deriv
                        if math.isfinite(x_newton):
                            f_newton = f(x_newton)
                            call_count += 1
                            if math.isfinite(f_newton):
                                if abs(f_newton) < abs(best_fx):
                                    best_x = x_newton
                                    best_fx = f_newton
                                if abs(f_newton) <= 1e-8:
                                    return x_newton
        except Exception:
            pass

    return best_x
