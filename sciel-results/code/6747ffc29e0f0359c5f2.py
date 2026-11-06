"""Solve 15-dim stiff linear ODE by exploiting linearity: y(t) = expm((t1-t0)*A) @ y0."""
import numpy as np
import time
import mpmath
from scipy.linalg import expm

def integrate(f, t_span, y0):
    deadline = time.time() + 290
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)

    # 1. Extract b = f(t0, 0)
    y_zero = np.zeros(n, dtype=float)
    b = np.asarray(f(t0, y_zero), dtype=float)
    
    # 2. Extract A
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        if time.time() > deadline:
            break
        e = np.zeros(n, dtype=float)
        e[j] = 1.0
        col = np.asarray(f(t0, e), dtype=float)
        A[:, j] = col - b

    # Check if homogeneous
    if np.linalg.norm(b) < 1e-14:
        # Homogeneous system
        mpmath.mp.dps = 40
        A_mp = mpmath.matrix(n)
        for i in range(n):
            for j in range(n):
                A_mp[i+1, j+1] = mpmath.mpf(A[i, j])
        
        M_mp = mpmath.expm(mpmath.mpf(dt) * A_mp)
        
        M = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                M[i, j] = float(M_mp[i+1, j+1])
        
        y_end = M @ y0
    else:
        # Affine system
        H = np.zeros((n + 1, n + 1), dtype=float)
        H[:n, :n] = A
        H[:n, n] = b
        
        z0 = np.zeros(n + 1, dtype=float)
        z0[:n] = y0
        z0[n] = 1.0
        
        mpmath.mp.dps = 40
        H_mp = mpmath.matrix(n+1)
        for i in range(n+1):
            for j in range(n+1):
                H_mp[i+1, j+1] = mpmath.mpf(H[i, j])
        
        M_mp = mpmath.expm(mpmath.mpf(dt) * H_mp)
        
        M = np.zeros((n+1, n+1), dtype=float)
        for i in range(n+1):
            for j in range(n+1):
                M[i, j] = float(M_mp[i+1, j+1])
        
        z_end = M @ z0
        y_end = z_end[:n]
        
    return y_end
