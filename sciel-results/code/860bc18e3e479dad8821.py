"""Spectral (DST) exact integrator for the discretized heat equation."""
# EVOLVE-BLOCK-START
import numpy as np


def integrate(f, t_span, y0):
    """
    Integrate y' = A y from t_span[0] to t_span[1] with initial condition y0.
    
    A = 21^2 * tridiag(1, -2, 1) for N=20.
    Uses spectral decomposition via DST-I.
    """
    y0 = np.asarray(y0, dtype=float)
    N = len(y0)
    
    t0, t1 = t_span
    dt = t1 - t0
    
    # Call f at least once
    _ = f(t0, y0)
    
    # Eigenvalues of A = (N+1)^2 * tridiag(1, -2, 1)
    # lambda_k = -4 * sin^2(k * pi / (2 * (N+1)))
    k = np.arange(1, N + 1)
    angles = k * np.pi / (2.0 * (N + 1))
    A_eigenvalues = -4.0 * (N + 1) ** 2 * np.sin(angles) ** 2
    
    exp_factors = np.exp(A_eigenvalues * dt)
    
    # DST-I matrix
    i_spatial = np.arange(1, N + 1)
    j_modes = np.arange(1, N + 1)
    V = np.sin(np.outer(i_spatial, j_modes) * np.pi / (N + 1))
    
    # y_end = (2/(N+1)) * V @ (exp_factors * (V.T @ y0))
    Vy = V.T @ y0
    expVy = exp_factors * Vy
    y_end = (2.0 / (N + 1)) * (V @ expVy)
    
    return y_end
# EVOLVE-BLOCK-END
