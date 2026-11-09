"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def _extract_matrix(f, t0, y0):
    """
    Extract the system matrix A using n calls.
    Strategy: Use y0 as the first basis vector and n-1 standard unit vectors for the rest.
    If this set is linearly dependent, fall back to standard unit vectors.
    """
    n = len(y0)
    
    # Attempt to use y0 and n-1 unit vectors
    # Basis vectors: v0 = y0, v1 = e0, ..., v_{n-1} = e_{n-2}
    V = np.zeros((n, n))
    V[:, 0] = y0
    for j in range(1, n):
        V[j-1, j] = 1.0  # e_{j-1} is at index j-1, column j
    
    # Check if V is well-conditioned / invertible
    # Use a small threshold for singularity
    try:
        # Check condition number or determinant magnitude
        # Using slogdet is more stable for checking singularity
        sign, logdet = np.linalg.slogdet(V)
        if sign == 0 or logdet < -30: # Threshold for near-singularity
            raise np.linalg.LinAlgError("Singular or ill-conditioned")
        
        # If good, use this basis
        F = np.zeros((n, n))
        F[:, 0] = f(t0, y0)
        for j in range(1, n):
            e = np.zeros(n)
            e[j-1] = 1.0
            F[:, j] = f(t0, e)
            
        A = F @ np.linalg.inv(V)
        return A
        
    except np.linalg.LinAlgError:
        # Fallback to standard unit vectors
        A = np.empty((n, n))
        for j in range(n):
            e = np.zeros(n)
            e[j] = 1.0
            A[:, j] = f(t0, e)
        return A


def _solve_linear_ode_explicit(A, t0, t1, y0):
    """
    Solve y' = A y using matrix exponential via scipy.
    """
    from scipy.linalg import expm
    dt = t1 - t0
    M = expm(A * dt)
    return M @ y0


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Extract A using n calls
    A = _extract_matrix(f, t0, y0)
    
    # Solve using matrix exponential
    y_end = _solve_linear_ode_explicit(A, t0, t1, y0)
    
    return y_end
# EVOLVE-BLOCK-END
