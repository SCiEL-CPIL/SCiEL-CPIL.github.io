# EVOLVE-BLOCK-START
import numpy as np


# Dormand-Prince 5(4) coefficients
_DP_A = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0 / 5, 0.0, 0.0, 0.0, 0.0, 0.0],
    [3.0 / 40, 9.0 / 40, 0.0, 0.0, 0.0, 0.0],
    [44.0 / 45, -56.0 / 15, 32.0 / 9, 0.0, 0.0, 0.0],
    [19372.0 / 6561, -25360.0 / 2187, 64448.0 / 6561, -212.0 / 729, 0.0, 0.0],
    [9017.0 / 3168, -355.0 / 33, 46732.0 / 5247, 49.0 / 176, -5103.0 / 18656, 0.0],
]
_DP_C = [0.0, 1.0 / 5, 3.0 / 10, 4.0 / 5, 8.0 / 9, 1.0]
_DP_B5 = [35.0 / 384, 0.0, 500.0 / 1113, 125.0 / 192, -2187.0 / 6784, 11.0 / 84, 0.0]
_DP_B4 = [5179.0 / 57600, 0.0, 7571.0 / 16695, 393.0 / 640, -92097.0 / 339200, 187.0 / 2100, 1.0 / 40]
_DP_E = np.array([
    71.0 / 57600, 0.0, -71.0 / 16695, 71.0 / 1920,
    -17253.0 / 339200, 22.0 / 525, -1.0 / 40,
])


def _dp_step(f, t, y, h):
    """One Dormand-Prince step. Returns (y5, y4, err, nfev).

    y5: 5th-order solution, y4: 4th-order solution, err: y5 - y4.
    """
    n = len(y)
    k = np.empty((7, n))
    k[0] = f(t, y)
    nfev = 1
    for i in range(1, 6):
        tmp = y + h * np.dot(_DP_A[i, :i], k[:i])
        k[i] = f(t + _DP_C[i] * h, tmp)
        nfev += 1
    y5 = y + h * np.dot(_DP_B5, k)
    y4 = y + h * np.dot(_DP_B4, k)
    tmp = y + h * np.dot(_DP_B4, k)
    k[6] = f(t + h, tmp)
    nfev += 1
    err = y5 - y4
    return y5, y4, err, nfev


def _safe_step_size(err_norm, tol, h, max_step, min_step):
    """Pick a safe next step size from the error estimate."""
    if err_norm <= 0.0:
        return min(max_step, h * 5.0)
    # error is O(h^5) for the 5th-order method -> scale by (tol/err)^(1/5)
    fac = (tol / err_norm) ** 0.2
    fac = min(5.0, max(0.1, fac))
    nh = h * fac
    if nh > max_step:
        nh = max_step
    if nh < min_step:
        nh = min_step
    return nh


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    total = t1 - t0
    if total == 0.0:
        # still must call f at least once
        f(t0, y)
        return y

    # tight tolerances to push global error near the 1e-14 floor
    atol = 1e-13
    rtol = 1e-11
    tol = atol + rtol * np.max(np.abs(y))

    max_step = abs(total) / 25.0
    min_step = abs(total) / 200000.0
    h = abs(total) / 200.0
    t = t0
    nfev = 0
    max_rejects = 400000

    while (t1 - t0 > 0 and t < t1) or (t1 - t0 < 0 and t > t1):
        if h > max_step:
            h = max_step
        if h < min_step:
            h = min_step
        # clamp so we do not overshoot the end
        if t1 - t0 > 0 and t + h > t1:
            h = t1 - t
        elif t1 - t0 < 0 and t + h < t1:
            h = t1 - t
        if h <= 0.0:
            break

        y5, y4, err, step_nfev = _dp_step(f, t, y, h)
        nfev += step_nfev
        err_norm = np.linalg.norm(err / (atol + rtol * np.abs(y5)))

        if err_norm <= 1.0:
            # accept
            y = y5
            t = t + h
            tol = atol + rtol * np.max(np.abs(y))
            h = _safe_step_size(err_norm, 1.0, h, max_step, min_step)
        else:
            # reject, shrink
            h = _safe_step_size(err_norm, 1.0, h, max_step, min_step)
            max_rejects -= 1
            if max_rejects <= 0:
                # bail out gracefully rather than run forever
                break

    return y
# EVOLVE-BLOCK-END
