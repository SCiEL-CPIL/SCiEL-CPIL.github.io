"""Fresh design: adaptive Radau IIA (order 5) with error control for stiff ODEs."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Radau IIA 5th-order (N=5) nodes on [0,1]
_NODES = np.array([
    0.0308536530354690,
    0.1383417242327530,
    0.3375953465707000,
    0.6624046534293000,
    0.8616582757672470,
    0.9691463469645310,
])
# b_i (6th-order weights), b_i' (5th-order weights)
_B6 = np.array([
    0.0245395439102813,
    0.0866449136520474,
    0.1855814177561270,
    0.1855814177561270,
    0.0866449136520474,
    0.0245395439102813,
])
_B5 = np.array([
    0.0222564810543630,
    0.0870037338824874,
    0.1868267745351120,
    0.1868267745351120,
    0.0870037338824874,
    0.0222564810543630,
])
# Stage coefficients a_ij
_A = np.array([
    [0.0009517284418903, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0083788488389724, 0.0037232500994608, 0.0, 0.0, 0.0, 0.0],
    [0.0295283383414107, 0.0283013600377452, 0.0095246198035111, 0.0, 0.0, 0.0],
    [0.0644453310807178, 0.0769289301733926, 0.0424508179261317, 0.0142696426675525, 0.0, 0.0],
    [0.1065470598853005, 0.1367621995829069, 0.0907266204025434, 0.0371942001734566, 0.0093574156441263, 0.0],
    [0.1569500218757900, 0.2078504535657611, 0.1485648206304000, 0.0730602745110606, 0.0270989784713144, 0.0060134550454308],
])


def _newton_solve(f, t, h, y, n, tol, maxit):
    """Solve y_new - h*f(t+h, y_new) = y for y_new via Newton with FD Jacobian."""
    y_new = y + h * f(t, y)  # explicit Euler predictor
    for _ in range(maxit):
        fz = f(t + h, y_new)
        resid = y_new - h * fz - y
        if np.linalg.norm(resid) <= tol * (1.0 + np.linalg.norm(y_new)):
            break
        J = np.empty((n, n))
        for j in range(n):
            e = np.zeros(n)
            scale = 1e-6 * max(1.0, abs(y_new[j]))
            e[j] = scale
            J[:, j] = (f(t + h, y_new + e) - fz) / scale
        A = np.eye(n) - h * J
        try:
            step = np.linalg.solve(A, resid)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(A, resid, rcond=None)[0]
        y_new = y_new - step
    return y_new


def _radau_step(f, t, h, y, n, rtol, atol, maxit):
    """One adaptive Radau IIA step. Returns (y_new, err_norm)."""
    # Precompute stage predictors
    Y = np.empty((6, n))
    Y[0] = y + h * _NODES[0] * f(t, y)
    for i in range(1, 6):
        s = y.copy()
        for j in range(i):
            s = s + h * _A[i, j] * Y[j]
        Y[i] = s
    # Newton solve each stage
    for i in range(6):
        Y[i] = _newton_solve(f, t, h, Y[i], n, 1e-12, maxit)
    # 6th-order solution
    y_new = y.copy()
    for i in range(6):
        y_new = y_new + h * _B6[i] * Y[i]
    # 5th-order solution (for error estimate)
    y_old = y.copy()
    for i in range(6):
        y_old = y_old + h * _B5[i] * Y[i]
    err = y_new - y_old
    scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
    err_norm = np.sqrt(np.sum((err / scale) ** 2)) / np.sqrt(n)
    return y_new, err_norm


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    t = t0
    h = 1e-3
    rtol = 1e-11
    atol = 1e-13
    maxit = 8
    start = time.time()
    deadline = start + 280.0
    best_y = y.copy()
    while t < t1:
        if time.time() > deadline:
            break
        h = min(h, t1 - t)
        y_new, err_norm = _radau_step(f, t, h, y, n, rtol, atol, maxit)
        if not np.all(np.isfinite(y_new)):
            h *= 0.5
            continue
        if err_norm < 1e-15:
            err_norm = 1e-15
        factor = min(5.0, max(0.2, 0.9 * err_norm ** (-0.2)))
        if err_norm <= 1.0:
            y = y_new
            t = t + h
            best_y = y.copy()
            h = h * factor
        else:
            h = h * factor
            if h < 1e-12:
                h = 1e-12
    return best_y
# EVOLVE-BLOCK-END
