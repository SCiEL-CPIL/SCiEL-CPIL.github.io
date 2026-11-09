"""
ODE solver for the 10x10 Heat Equation using Kronecker Structure Exploitation.
"""
import numpy as np
from scipy.linalg import expm

def integrate(f, t_span, y0):
    """
    Integrates y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Exploits the fact that the 2D heat equation on a square grid is separable:
    The system matrix A can be written as I ⊗ L + L ⊗ I, where L is the 1D
    Laplacian. We determine L by a single probe call to f on the identity matrix,
    then compute the solution using the matrix exponential of the smaller 1D operator.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # The grid is 10x10, so n should be 100.
    # We assume a square grid of size sqrt(n) x sqrt(n).
    side = int(np.sqrt(n))
    if side * side != n:
        # Fallback to general method if not square (though task specifies 10x10)
        A = np.zeros((n, n), dtype=float)
        for j in range(n):
            e_j = np.zeros(n, dtype=float)
            e_j[j] = 1.0
            A[:, j] = f(t0, e_j)
        dt = t1 - t0
        M = expm(A * dt)
        return M @ y0

    # Reshape y0 to 2D grid (C-order: row-major)
    Y0 = y0.reshape(side, side)
    
    # Probe the system to find the 1D operator L.
    # Let I_side be the identity matrix of size 'side'.
    # The vectorized identity corresponds to the matrix I_side.
    # f(t, vec(I)) = vec(L I + I L^T) = vec(L + L^T).
    # Assuming L is symmetric (standard for heat equation), L + L^T = 2L.
    I_side = np.eye(side)
    probe_vec = I_side.flatten()
    R = f(t0, probe_vec)
    M_probe = R.reshape(side, side)
    
    # Extract L
    L = 0.5 * M_probe
    
    # Compute the 1D exponential
    dt = t1 - t0
    E = expm(L * dt)
    
    # The solution to Y' = L Y + Y L^T is Y(t) = E Y(0) E^T
    Y_end = E @ Y0 @ E.T
    
    # Flatten back to vector
    y_end = Y_end.flatten()
    
    return y_end
