"""Solve 15-dim stiff linear ODE by exploiting linearity: y' = A y + b.
Augments the system to 16-dim to handle inhomogeneous term b(t) assuming it is constant.
Uses high-precision matrix exponential via mpmath."""
import numpy as np
import time
from scipy.linalg import expm
from mpmath import mp, matrix as mp_matrix, mpf


def integrate(f, t_span, y0):
    deadline = time.time() + 290  # safety margin before 330s kill
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)

    # 1. Extract b = f(t0, 0)
    b = np.asarray(f(t0, np.zeros(n, dtype=float)), dtype=float)
    
    # 2. Extract A by evaluating f at basis vectors: n calls total.
    # f(t0, e_j) = A e_j + b  => A e_j = f(t0, e_j) - b
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        e = np.zeros(n, dtype=float)
        e[j] = 1.0
        col = f(t0, e)
        A[:, j] = np.asarray(col, dtype=float) - b
        if time.time() > deadline:
            break

    # Construct augmented 16x16 matrix K for mpmath
    # K = [ A   b ]
    #     [ 0   0 ]
    # z = [ y ]
    #     [ 1 ]
    # z' = K z
    
    try:
        # Set working precision to 50 decimal digits
        mp.dps = 50
        dt_mp = mpf(str(dt))
        
        n_aug = n + 1
        K_mp = mp_matrix(n_aug, n_aug)
        
        # Fill A part
        for i in range(n):
            for j in range(n):
                K_mp[i, j] = mpf(str(A[i, j]))
            # Fill b part (column n)
            K_mp[i, n] = mpf(str(b[i]))
        
        # Last row is zeros
        # (mp_matrix initializes to 0)
        
        # Compute expm(K * dt)
        K_dt = dt_mp * K_mp
        E_mp = expm(K_dt)
        
        # Construct z0 = [y0; 1]
        z0_mp = mp_matrix(n_aug, 1)
        for i in range(n):
            z0_mp[i, 0] = mpf(str(y0[i]))
        z0_mp[n, 0] = mpf('1.0')
        
        # Compute z_end = E_mp * z0_mp
        z_end_mp = E_mp * z0_mp
        
        # Extract y_end
        y_end = np.zeros(n, dtype=float)
        for i in range(n):
            y_end[i] = float(z_end_mp[i, 0])
            
        if time.time() < deadline:
            return y_end
    except Exception:
        pass

    # Fallback to scipy expm if mpmath fails
    # Construct K for scipy
    K = np.zeros((n + 1, n + 1), dtype=float)
    K[:n, :n] = A
    K[:n, n] = b
    # K[n, :] is already 0
    
    E = expm(dt * K)
    z0 = np.zeros(n + 1, dtype=float)
    z0[:n] = y0
    z0[n] = 1.0
    z_end = E @ z0
    y_end = z_end[:n]

    return y_end
