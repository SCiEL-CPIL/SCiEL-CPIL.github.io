"""Rewritten ODE integrator for the fixed Lorenz IVP using a 5(4) adaptive RKF method."""
import numpy as np


def _lorenz_deriv(t, y):
    # Fallback (unused if f is provided correctly)
    y0, y1, y2 = y
    return np.array([10.0 * (y1 - y0),
                     y0 * (28.0 - y2) - y1,
                     y0 * y1 - (8.0 / 3.0) * y2])


# Ralston 5th-order coefficients (Fehlberg 5, RKF5)
_RKF5_C = np.array([0.0, 1.0 / 4.0, 3.0 / 8.0, 12.0 / 13.0, 1.0, 1.0 / 2.0])
_RKF5_A = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0 / 4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [3.0 / 32.0, 9.0 / 32.0, 0.0, 0.0, 0.0, 0.0],
    [1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0, 0.0, 0.0, 0.0],
    [439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0, 0.0, 0.0],
    [-8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0, 0.0],
])
_RKF5_B5 = np.array([25.0 / 216.0, 0.0, 1408.0 / 2565.0, 2197.0 / 4104.0, -1.0 / 5.0, 0.0])
_RKF5_B4 = np.array([16.0 / 135.0, 0.0, 6656.0 / 12825.0, 28561.0 / 56430.0, -9.0 / 50.0, 2.0 / 55.0])


def _rkf5_step(f, t, y, h):
    """One 5(4) RKF step. Returns y5 (5th order), y4 (4th order), error estimate."""
    c = _RKF5_C
    A = _RKF5_A
    b5 = _RKF5_B5
    b4 = _RKF5_B4

    k = np.empty((6, len(y)))
    for i in range(6):
        yi = y.copy()
        for j in range(i):
            yi = yi + h * A[i, j] * k[j]
        k[i] = f(t + c[i] * h, yi)

    y5 = y + h * np.dot(b5, k)
    y4 = y + h * np.dot(b4, k)
    err = y5 - y4
    return y5, y4, err


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    h = (t1 - t0) / 1000.0
    t = t0
    rtol = 1e-11
    atol = 1e-13
    max_steps = 100000
    min_h = (t1 - t0) / 1e7
    max_h = (t1 - t0) / 10.0

    for _ in range(max_steps):
        if t >= t1 - 1e-15:
            break
        # Don't overshoot t1
        h_use = min(h, t1 - t)

        y5, y4, err = _rkf5_step(f, t, y, h_use)

        # Error norm: scaled
        scale = np.maximum(np.abs(y), np.abs(y5))
        err_norm = np.linalg.norm(err / (atol + rtol * scale)) / np.sqrt(n)

        if err_norm <= 1.0:
            # Accept step
            y = y5
            t = t + h_use
            if err_norm > 0:
                h_new = h_use * min(5.0, max(0.1, 0.9 * err_norm ** (-0.2)))
            else:
                h_new = h_use * 5.0
            h = min(max(h_new, min_h), max_h)
        else:
            # Reject step, reduce h
            h = h_use * max(0.1, 0.9 * err_norm ** (-0.25))
            h = max(h, min_h)

    return y
