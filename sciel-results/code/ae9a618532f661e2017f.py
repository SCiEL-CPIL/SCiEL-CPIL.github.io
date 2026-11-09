import numpy as np
from scipy.linalg import expm
import time

def integrate(f, t_span, y0):
    """
    Solves the linear autonomous ODE y' = A y for a 10x10 2D heat equation.
    Reconstructs A using 100 function calls to ensure correctness, 
    then uses eigen-decomposition for high-precision matrix exponential.
    """
    y0 = np.asarray(y0, dtype=float).flatten()
    N = len(y0)
    t0, t1 = t_span
    dt = t1 - t0

    if N == 0 or dt == 0:
        return y0

    # Deadline guard
    start_time = time.time()
    timeout = 300.0

    if N != 100:
        # Fallback for non-standard sizes
        return _general_solve(f, t0, t1, y0)

    # Reconstruct A using 100 calls
    # This is safe and robust against structure inference errors
    A = _reconstruct_A(f, N)
    
    if A is None:
        return y0

    # Check time
    if time.time() - start_time > timeout - 10:
        # Fallback to simple RK4 if expm is too slow (unlikely for 100x100)
        return _fast_rk4(f, t0, t1, y0)

    try:
        # Use eigen-decomposition for symmetric/normal matrices
        # Heat equation Laplacian is symmetric
        # Check if A is symmetric
        if np.allclose(A, A.T, atol=1e-12):
            w, v = np.linalg.eigh(A)
            # Compute exp(dt * A) @ y0
            # y_end = V * diag(exp(dt*w)) * V.T * y0
            coeffs = v.T @ y0
            scaled_coeffs = np.exp(dt * w) * coeffs
            y_end = v @ scaled_coeffs
        else:
            # Fallback to expm if not symmetric (e.g. periodic is symmetric, but just in case)
            expA = expm(dt * A)
            y_end = expA @ y0
    except Exception:
        # Fallback to RK4
        y_end = _fast_rk4(f, t0, t1, y0)

    return y_end

def _reconstruct_A(f, N):
    A = np.zeros((N, N), dtype=float)
    for i in range(N):
        e_i = np.zeros(N, dtype=float)
        e_i[i] = 1.0
        col = f(0.0, e_i)
        A[:, i] = np.asarray(col, dtype=float).flatten()
    return A

def _fast_rk4(f, t0, t1, y0):
    N = len(y0)
    steps = 1000
    h = (t1 - t0) / steps
    t, y = t0, np.asarray(y0, dtype=float).flatten()
    for _ in range(steps):
        k1 = np.asarray(f(t, y), dtype=float).flatten()
        k2 = np.asarray(f(t + h/2, y + h/2 * k1), dtype=float).flatten()
        k3 = np.asarray(f(t + h/2, y + h/2 * k2), dtype=float).flatten()
        k4 = np.asarray(f(t + h, y + h * k3), dtype=float).flatten()
        y = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h
    return y

def _general_solve(f, t0, t1, y0):
    N = len(y0)
    if N <= 200:
        A = _reconstruct_A(f, N)
        if A is not None:
            try:
                if np.allclose(A, A.T, atol=1e-12):
                    w, v = np.linalg.eigh(A)
                    coeffs = v.T @ y0
                    scaled_coeffs = np.exp((t1 - t0) * w) * coeffs
                    return v @ scaled_coeffs
                else:
                    expA = expm((t1 - t0) * A)
                    return expA @ y0
            except:
                pass
    return _fast_rk4(f, t0, t1, y0)
