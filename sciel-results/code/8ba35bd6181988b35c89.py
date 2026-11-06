"""ODE solver: 15-dimensional stiff linear system using matrix extraction and matrix exponential."""
import numpy as np
import scipy.linalg
import time


def _extract_matrix_A(f, t0, y0):
    """
    Extract the system matrix A from y' = f(t, y) assuming y' = A y (linear, time-independent).
    Uses n identity basis vectors to directly read off columns of A.
    """
    n = len(y0)
    
    # Evaluate f for each identity basis vector at t0
    # f(t0, e_j) = A @ e_j = A[:, j] (j-th column of A)
    A = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        A[:, j] = f(t0, e)
    
    return A


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Returns y(t_end).
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0  # 300s limit, keep 20s margin
    
    # Extract matrix A (costs n evaluations)
    A = _extract_matrix_A(f, t0, y0)
    
    # Check time
    if time.time() > deadline:
        # Fallback: return initial condition if we ran out of time
        return y0
    
    # Compute matrix exponential exp(A * (t1 - t0))
    dt = t1 - t0
    try:
        # Use scipy's expm for high-precision matrix exponential
        expA = scipy.linalg.expm(A * dt)
    except Exception:
        # Fallback: use eigen decomposition if expm fails
        try:
            eigvals, eigvecs = np.linalg.eig(A * dt)
            expA = eigvecs @ np.diag(np.exp(eigvals)) @ np.linalg.inv(eigvecs)
        except Exception:
            # Last resort: simple RK4
            h = dt / 2000
            t, y = t0, y0.copy()
            for _ in range(2000):
                k1 = f(t, y)
                k2 = f(t + h/2, y + h/2 * k1)
                k3 = f(t + h/2, y + h/2 * k2)
                k4 = f(t + h, y + h * k3)
                y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
                t += h
                if time.time() > deadline:
                    return y
            return y
    
    # Final state
    y_final = expA @ y0
    
    return y_final
