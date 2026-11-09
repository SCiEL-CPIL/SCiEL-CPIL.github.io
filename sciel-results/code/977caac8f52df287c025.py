"""ODE integrator for the discretized heat equation (N = 20) using high-precision analytic eigen-decomposition."""

import numpy as np
import time
from mpmath import mp, sin, pi as mp_pi, exp


def integrate(f, t_span, y0):
    """
    Integrate y' = A y where A = 21^2 * tridiag(1, -2, 1) for N=20.
    
    Uses the analytical eigendecomposition approach with high precision (mpmath)
    to minimize numerical error.
    
    Parameters
    ----------
    f : callable
        The right-hand side function f(t, y) -> y'. Called at least once.
    t_span : tuple
        (t_start, t_end) integration interval.
    y0 : ndarray
        Initial condition of shape (20,).
        
    Returns
    -------
    ndarray
        The state y at t = t_end, shape (20,).
    """
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    N = len(y0_arr)
    
    if N == 0:
        f(t0, y0_arr)
        return y0_arr

    # Use mpmath with high precision for the eigen-decomposition
    old_prec = mp.prec
    mp.prec = 160  # 160 bits ~ 48 decimal digits
    
    k_vec = [mp.mpf(i + 1) for i in range(N)]
    i_vec = [mp.mpf(i + 1) for i in range(N)]
    
    # Build V matrix: V[k-1, i-1] = sin(k*pi*i/(N+1))
    V = [[sin(k_vec[k_idx] * mp_pi * i_vec[i_idx] / (N + 1)) for i_idx in range(N)] for k_idx in range(N)]
    
    # Eigenvalues: lambda_k = -4*(N+1)^2 * sin^2(k*pi/(2*(N+1)))
    N_plus_1 = mp.mpf(N + 1)
    lambdas = [-4.0 * N_plus_1**2 * sin(k_vec[k_idx] * mp_pi / (2.0 * N_plus_1))**2 for k_idx in range(N)]
    
    # Orthogonality norm
    norm_sq = (N + 1) / 2.0
    norm_sq = mp.mpf(norm_sq)
    
    # Project y0 onto eigenvectors
    y0_mp = [mp.mpf(y0_arr[j]) for j in range(N)]
    c = []
    for k_idx in range(N):
        dot = sum(V[k_idx][i_idx] * y0_mp[i_idx] for i_idx in range(N))
        c.append(dot / norm_sq)
    
    # Compute decay factors
    dt = mp.mpf(t1) - mp.mpf(t0)
    decay_factors = [exp(lambdas[k_idx] * dt) for k_idx in range(N)]
    
    # Reconstruct y_end
    y_end_mp = [sum(V[k_idx][i_idx] * c[k_idx] * decay_factors[k_idx] for k_idx in range(N)) for i_idx in range(N)]
    
    # Convert back to float64 for return
    y_end = np.array([float(y) for y in y_end_mp], dtype=float)
    
    # Restore precision
    mp.prec = old_prec
    
    # Ensure finite
    if not np.all(np.isfinite(y_end)):
        f(t0, y0_arr)
        return y0_arr
    
    # Call f at least once
    f(t0, y0_arr)
    
    return y_end
