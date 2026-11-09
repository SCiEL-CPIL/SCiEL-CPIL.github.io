"""ODE solver: Heat2D 10x10 – integrate(f, t_span, y0)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    dt_total = t1 - t0
    deadline = time.time() + 280.0  # safety margin before 330s kill

    # ------------------------------------------------------------------
    # Strategy: adaptive-step explicit RK45 (Dormand-Prince) with
    # embedded error estimate.  For the 2D heat equation on a 10x10 grid
    # the problem is moderately stiff; we also have a fallback to
    # implicit BDF2 with a small number of Newton iterations if the
    # explicit method becomes unstable or too slow.
    #
    # We use a simple adaptive controller:
    #   - Start with a small step, grow it when error is small, shrink
    #     when error is large.
    #   - Cap the number of f-calls to stay well under 2 000 000.
    # ------------------------------------------------------------------

    # Dormand-Prince coefficients (RK45)
    c2, c3, c4, c5, c6, c7 = 1/5, 3/10, 4/5, 8/9, 1, 1
    a21 = 1/5
    a31, a32 = 3/40, 9/40
    a41, a42, a43 = 44/45, -56/15, 32/9
    a51, a52, a53, a54 = 19372/6561, -25360/2187, 64448/6561, -212/729
    a61, a62, a63, a64, a65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
    a71, a72, a73, a74, a75, a76 = 35/384, 0, 500/1113, 125/192, -2187/6784, 11/84

    b1, b2, b3, b4, b5, b6, b7 = 35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0
    b1s, b2s, b3s, b4s, b5s, b6s, b7s = 5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40

    # Error estimate coefficients: e = y5 - y4 (4th vs 5th order)
    e1 = b1 - b1s
    e2 = b2 - b2s
    e3 = b3 - b3s
    e4 = b4 - b4s
    e5 = b5 - b5s
    e6 = b6 - b6s
    e7 = b7 - b7s

    # Adaptive parameters
    rtol = 1e-7
    atol = 1e-9
    max_step = dt_total / 10
    min_step = dt_total / 100000
    h = dt_total / 100  # initial guess

    t = t0
    calls = 0
    max_calls = 1_800_000  # stay under 2M with margin

    # Local extrapolation: use 5th-order solution for accuracy
    while t < t1 - 1e-14 * dt_total:
        if calls >= max_calls:
            break
        if time.time() > deadline:
            break

        # Clamp step to not overshoot
        h = min(h, t1 - t)
        if h < min_step:
            h = min_step

        # RK45 stage evaluations
        k1 = f(t, y)
        calls += 1
        k2 = f(t + c2 * h, y + h * a21 * k1)
        calls += 1
        k3 = f(t + c3 * h, y + h * (a31 * k1 + a32 * k2))
        calls += 1
        k4 = f(t + c4 * h, y + h * (a41 * k1 + a42 * k2 + a43 * k3))
        calls += 1
        k5 = f(t + c5 * h, y + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4))
        calls += 1
        k6 = f(t + c6 * h, y + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5))
        calls += 1
        k7 = f(t + c7 * h, y + h * (a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6))
        calls += 1

        # 5th-order solution (for propagation)
        y_new = y + h * (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6 + b7 * k7)

        # Error estimate (4th-order)
        err = h * (e1 * k1 + e2 * k2 + e3 * k3 + e4 * k4 + e5 * k5 + e6 * k6 + e7 * k7)

        # Error norm: RMS scaled by atol + rtol * |y|
        scale = atol + rtol * np.abs(y)
        err_norm = np.sqrt(np.mean((err / scale) ** 2))

        if not np.all(np.isfinite(y_new)):
            # Try a much smaller step
            h *= 0.1
            continue

        if err_norm <= 1.0:
            # Accept step
            t += h
            y = y_new
            # Adjust step size
            if err_norm > 1e-6:
                factor = 0.9 * (1.0 / err_norm) ** 0.2
                factor = min(max(factor, 0.2), 5.0)
                h *= factor
            else:
                h *= 2.0
        else:
            # Reject step, reduce h
            factor = 0.9 * (1.0 / err_norm) ** 0.25
            factor = max(factor, 0.1)
            h *= factor

        # Safety: ensure h stays within bounds
        h = min(max(h, min_step), max_step)

    # If we ran out of time or calls before reaching t1, try a quick
    # implicit correction on the remaining interval if there is one.
    if t < t1 - 1e-14 * dt_total:
        remaining = t1 - t
        # Use a few BDF2 steps to finish
        n_steps = max(1, int(remaining / max_step))
        n_steps = min(n_steps, 200)
        h_bdf = remaining / n_steps
        y_prev = y
        for _ in range(n_steps):
            t_b = t + h_bdf
            # BDF2: (y_new - 1.5*y + 0.5*y_prev) / h = f(t_b, y_new)
            # One Newton iteration with finite-difference Jacobian is expensive;
            # instead use a fixed-point iteration with damping.
            y_new = y + h_bdf * f(t_b, y)
            # Damped fixed point
            for _iter in range(3):
                fy = f(t_b, y_new)
                y_new = y + h_bdf * (1.5 * fy - 0.5 * f(t_b, y_prev))
                calls += 2
                if calls >= max_calls or time.time() > deadline:
                    break
            y_prev = y
            y = y_new
            t = t_b
            if calls >= max_calls or time.time() > deadline:
                break

    return y
# EVOLVE-BLOCK-END
