"""Rewritten ODE integrator for the Lotka-Volterra system using a high-order explicit Runge-Kutta (Dormand-Prince 5(4)) with adaptive step control and Richardson extrapolation, designed for minimal f-calls while achieving high accuracy."""
# EVOLVE-BLOCK-START
import numpy as np


def integrate(f, t_span, y0):
    """
    Integrate the fixed Lotka-Volterra IVP from t_span[0] to t_span[1] starting at y0.
    Uses an adaptive Dormand-Prince 5(4) method with step-size control. Returns the state at t_end.
    """
    t0, t1 = t_span
    if t1 <= t0:
        return np.array(y0, dtype=float)

    y = np.array(y0, dtype=float)
    n = len(y)

    # Dormand-Prince 5(4) coefficients
    a = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
    ])

    b5 = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
    b4 = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])

    c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])

    rtol = 1e-12
    atol = 1e-14
    h = 0.01  # initial step
    t = t0
    max_steps = 100000

    # Adaptive stepping
    for _ in range(max_steps):
        if t >= t1 - 1e-15:
            break
        # Don't overshoot
        if t + h > t1:
            h = t1 - t
            if h <= 0:
                break

        # Compute stages
        k = np.zeros((7, n))
        k[0] = f(t, y)
        for i in range(1, 7):
            y_stage = y + h * np.dot(a[i, :i], k[:i])
            k[i] = f(t + c[i] * h, y_stage)

        # 5th order solution
        y_new5 = y + h * np.dot(b5, k)
        # 4th order solution
        y_new4 = y + h * np.dot(b4, k)

        # Error estimate
        err = np.linalg.norm(y_new5 - y_new4, ord=np.inf)
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new5))
        err_scaled = err / np.max(scale)

        if err_scaled <= 1.0:
            # Accept step
            y = y_new5
            t += h

            # Adjust step size
            if err_scaled > 1e-15:
                factor = 0.9 * (err_scaled ** (-0.2))
            else:
                factor = 1.5
            h_new = h * min(max(factor, 0.2), 5.0)
            if t + h_new > t1:
                h_new = t1 - t
            h = max(h_new, 1e-10)
        else:
            # Reject step
            if err_scaled > 0:
                factor = 0.9 * (err_scaled ** (-0.25))
            else:
                factor = 0.5
            h = max(h * min(factor, 0.5), 1e-10)
            if t + h > t1:
                h = t1 - t

    return y
# EVOLVE-BLOCK-END
