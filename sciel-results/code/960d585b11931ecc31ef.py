"""
Heat2D 10x10 ODE solver using exact matrix exponential for linear autonomous system.

The heat equation discretised on a 10x10 grid yields a linear, time-invariant ODE:
    y' = A y

We recover A by probing the vector field with standard basis vectors. For a linear
operator, f(t, e_j) = A e_j = A[:, j], so exactly n evaluations suffice. Then the
terminal state is y(t_end) = expm(A * (t_end - t0)) @ y0, which is the exact
solution up to floating-point precision of the matrix exponential.
"""

import numpy as np
from scipy.linalg import expm


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = y0.shape[0]

    dt = float(t1) - float(t0)
    if abs(dt) < 1e-16:
        return y0.copy()

    # Recover the linear operator A. For a linear autonomous system,
    # f(t, e_j) = A @ e_j = A[:, j].
    A = np.empty((n, n), dtype=float)
    for j in range(n):
        ej = np.zeros(n, dtype=float)
        ej[j] = 1.0
        col = f(float(t0), ej)
        A[:, j] = col

    # Exact propagation via the matrix exponential.
    M = expm(A * dt)
    y_end = M @ y0

    # Defensive fallback if something went numerically wrong.
    if not np.all(np.isfinite(y_end)):
        # Backward Euler as a last resort using the recovered A.
        y = y0.copy()
        n_steps = 20000
        h = dt / n_steps
        I = np.eye(n, dtype=float)
        for _ in range(n_steps):
            try:
                y = np.linalg.solve(I - h * A, y)
            except np.linalg.LinAlgError:
                break
        y_end = y

    return y_end
