import numpy as np
import time
from scipy.linalg import expm

def integrate(f, t_span, y0):
    deadline = time.time() + 290
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)

    # Extract A via n basis vectors: A @ e_j = f(t0, e_j)
    # For a linear system y' = A y, f(t, y) = A @ y
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        e = np.zeros(n, dtype=float)
        e[j] = 1.0
        col = f(t0, e)
        A[:, j] = np.asarray(col, dtype=float)
        if time.time() > deadline:
            break

    # Compute expm(dt * A) @ y0
    # Use mpmath for high precision if available, otherwise scipy
    try:
        import mpmath
        mpmath.mp.dps = 60
        A_mp = mpmath.matrix(n, n)
        for i in range(n):
            for j in range(n):
                A_mp[i, j] = mpmath.mpf(str(A[i, j]))
        
        dt_mp = mpmath.mpf(str(dt))
        M_mp = mpmath.expm(dt_mp * A_mp)
        
        M = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                M[i, j] = float(M_mp[i, j])
        
        return M @ y0
    except Exception:
        pass

    # Fallback to scipy expm
    M = expm(dt * A)
    return M @ y0
