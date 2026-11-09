"""ODE solver: Heat2D 10x10 - using exact linear solution via spectral decomposition."""
import numpy as np
from scipy.linalg import expm
import time

# Pre-compute the sine transform matrix for the 10x10 grid
# The eigenvectors of the 1D Laplacian with Dirichlet BCs are sine functions
N = 10
# j, k from 1 to 10
j = np.arange(1, N + 1)
k = np.arange(1, N + 1)
# V[j-1, k-1] = sin(j * k * pi / (N + 1))
V = np.sin(np.outer(j, k) * np.pi / (N + 1))
# Normalize the rows to be orthonormal? 
# The matrix V is orthogonal up to a scaling factor.
# Actually, V.T @ V = (N+1)/2 * I.
# So V_inv = 2/(N+1) * V.T
# We can just use V.T and V with appropriate scaling in the transform.
# Or simply normalize V to be orthonormal.
norms = np.sqrt(np.sum(V**2, axis=0))
V_orth = V / norms
V_orth_T = V_orth.T

def integrate(f, t_span, y0):
    """Solve y' = f(t, y) for the 10x10 heat equation."""
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin for 300s limit
    
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float).flatten()
    n = len(y0)
    
    if n != 100:
        return _rk4(f, t0, t1, y0, 10000)
    
    dt = t1 - t0
    if dt == 0:
        return y0

    # 1. Try 1-call spectral calibration
    f_val = f(0.0, y0)
    f_val = np.asarray(f_val, dtype=float).flatten()
    
    # Construct unscaled Laplacian A_unscaled = kron(I, T) + kron(T, I)
    # T = tridiag(-1, 2, -1)
    T = 2.0 * np.eye(N) - 1.0 * (np.eye(N, k=1) + np.eye(N, k=-1))
    I_N = np.eye(N)
    A_unscaled = np.kron(I_N, T) + np.kron(T, I_N)
    
    Ay = A_unscaled @ y0
    denom = np.dot(Ay, Ay)
    
    if denom < 1e-30:
        # y0 is in the null space (unlikely for heat eq) or zero
        return y0

    scale = np.dot(f_val, Ay) / denom
    residual = np.linalg.norm(f_val - scale * Ay) / max(1.0, np.linalg.norm(f_val))
    
    if residual < 1e-10:
        # Use exact spectral solution
        # A = scale * A_unscaled
        # Eigenvalues of A_unscaled are lambda_i + lambda_j where lambda are eigenvalues of T
        # Eigenvalues of T: 2 - 2*cos(k*pi/(N+1))
        eigvals_T = 2.0 - 2.0 * np.cos(np.arange(1, N + 1) * np.pi / (N + 1))
        eigsum = eigvals_T[:, None] + eigvals_T[None, :]
        
        # Transform y0 to eigenbasis
        # Y0 = y0.reshape(N, N)
        # Y0_eig = V_orth.T @ Y0 @ V_orth
        Y0 = y0.reshape(N, N)
        Y0_eig = V_orth_T @ Y0 @ V_orth
        
        # Apply exponential
        # exp(dt * scale * eigsum)
        multipliers = np.exp(dt * scale * eigsum)
        Y_end_eig = Y0_eig * multipliers
        
        # Transform back
        Y_end = V_orth @ Y_end_eig @ V_orth_T
        y_end = Y_end.flatten()
        
        # Verify time limit
        if time.time() > deadline:
            return y_end
        return y_end

    # 2. Fallback: Extract full matrix A using 100 calls
    # This is robust if the structure is slightly different (e.g. different BCs or weights)
    A_cols = []
    for i in range(n):
        if time.time() > deadline:
            return _rk4(f, t0, t1, y0, 10000)
        e_i = np.zeros(n)
        e_i[i] = 1.0
        col_i = f(0.0, e_i)
        col_i = np.asarray(col_i, dtype=float).flatten()
        A_cols.append(col_i)
        
    A = np.column_stack(A_cols)
    
    # Check if A is consistent with f(0, y0)
    # f(0, y0) should be A @ y0
    f_check = A @ y0
    residual2 = np.linalg.norm(f_val - f_check) / max(1.0, np.linalg.norm(f_val))
    
    if residual2 < 1e-6:
        # Use expm for exact solution
        try:
            expA = expm(dt * A)
            y_end = expA @ y0
            if time.time() > deadline:
                return y_end
            return y_end
        except Exception:
            pass

    # 3. Fallback: High-step RK4
    return _rk4(f, t0, t1, y0, 10000)

def _rk4(f, t0, t1, y0, steps):
    """Fallback RK4 solver."""
    h = (t1 - t0) / steps
    t, y = t0, np.asarray(y0, dtype=float).flatten()
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y
