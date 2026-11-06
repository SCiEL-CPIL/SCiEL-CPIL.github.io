"""Solve 15-dim stiff linear ODE by exploiting linearity: y(t) = expm((t1-t0)*A) @ y0.
Uses robust extraction of A and high-precision matrix exponential via
scipy.linalg.expm, with fallback to eigenvalue method when well-conditioned."""
import numpy as np
import time
from scipy.linalg import expm, eig


def integrate(f, t_span, y0):
    deadline = time.time() + 290  # safety margin before 330s kill
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)

    # For a linear system y' = A @ y, f(t, y) = A @ y.
    # Extract A by evaluating f at basis vectors: n calls total.
    
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        e = np.zeros(n, dtype=float)
        e[j] = 1.0
        col = f(t0, e)
        A[:, j] = np.asarray(col, dtype=float)
        if time.time() > deadline:
            break

    # Compute matrix exponential: y(t1) = expm(dt * A) @ y0
    # Try eigenvalue-based method if well-conditioned for potentially better accuracy
    use_eig = False
    try:
        vals, vecs = eig(dt * A)
        cond = np.linalg.cond(vecs)
        if cond < 1e8:
            use_eig = True
    except Exception:
        use_eig = False

    if use_eig:
        # expm(dt*A) = V @ diag(exp(vals)) @ V^{-1}
        try:
            V_inv = np.linalg.inv(vecs)
            M = (vecs @ np.diag(np.exp(vals))) @ V_inv
        except Exception:
            M = expm(dt * A)
    else:
        M = expm(dt * A)
    
    y_end = M @ y0

    return y_end
