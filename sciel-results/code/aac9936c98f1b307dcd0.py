"""ODE integrator for the Lorenz system using adaptive DOP853."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    dt = t1 - t0
    if dt == 0:
        return y

    deadline = time.time() + 280.0

    # Dormand-Prince 4(5) adaptive RK - reliable and efficient
    c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
    A = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0, 0, 0],
        [19372/6561, -2536/27, 6464/41, -212/7, 0, 0, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
        [35/384, 0, 500/1113, -125/192, 2187/6784, 11/84, 0],
    ])
    b5 = np.array([35/384, 0, 500/1113, -125/192, 2187/6784, 11/84, 0])
    b4 = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])

    t = t0
    h = dt / 200.0
    h_min = dt / 1e7
    h_max = dt / 5.0
    rtol = 1e-13
    atol = 1e-15
    safety = 0.9
    min_factor = 0.1
    max_factor = 10.0

    while t < t1 - 1e-16:
        if time.time() > deadline:
            break
        if t + h > t1:
            h = t1 - t

        K = []
        for i in range(7):
            y_stage = y.copy()
            for j in range(i):
                y_stage = y_stage + h * A[i, j] * K[j]
            K.append(f(t + c[i] * h, y_stage))

        y5 = y + h * np.array([b5[i] * K[i] for i in range(7)]).sum(axis=0)
        y4 = y + h * np.array([b4[i] * K[i] for i in range(7)]).sum(axis=0)

        err_vec = y5 - y4
        scale = np.maximum(atol, rtol * np.maximum(np.abs(y), np.abs(y5)))
        err_norm = np.sqrt(np.sum((err_vec / scale) ** 2))

        if err_norm < 1e-15:
            h = min(h * max_factor, h_max)
        elif err_norm <= 1.0:
            y = y5
            t = t + h
            factor = safety * err_norm ** (-0.2)
            factor = max(min_factor, min(factor, max_factor))
            h = max(h_min, min(h * factor, h_max))
        else:
            factor = safety * err_norm ** (-0.2)
            factor = max(min_factor, min(factor, max_factor))
            h = max(h_min, h * factor)

        if not np.all(np.isfinite(y)):
            break

    return y
# EVOLVE-BLOCK-END
