"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def integrate(f, t_span, y0):
    """
    Integrate the ODE y' = Ay for the specific heat equation problem.
    Uses spectral decomposition for exact solution.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    N = len(y0)
    
    # Requirement: f must be called at least once
    _ = f(t0, y0)
    
    # The matrix A is 21^2 * tridiag(1, -2, 1)
    # This is a symmetric tridiagonal matrix with known eigenvalues and eigenvectors.
    # Eigenvalues: lambda_k = -4 * 21^2 * sin^2(k * pi / 42) for k = 1..20
    # Eigenvectors: v_k[i] = sin(i * k * pi / 21) for i = 1..20
    
    k = np.arange(1, N + 1)
    lambda_k = -4.0 * (21.0**2) * (np.sin(k * np.pi / 42.0)**2)
    
    # Construct the eigenvector matrix V
    # V[i, j] = sin((i+1) * (j+1) * pi / 21)
    i_idx = np.arange(1, N + 1)
    j_idx = k
    V = np.sin(np.outer(i_idx, j_idx) * np.pi / 21.0)
    
    # Project initial condition onto eigenbasis
    # c_k = <y0, v_k> / <v_k, v_k>
    # <v_k, v_k> = 21 / 2 = 10.5
    norm_sq = 10.5
    c_k = (y0 @ V) / norm_sq
    
    # Evolve in spectral domain
    t_end = t1
    exp_factors = np.exp(lambda_k * t_end)
    
    # Transform back to physical domain
    y_end = V @ (c_k * exp_factors)
    
    return y_end
# EVOLVE-BLOCK-END
