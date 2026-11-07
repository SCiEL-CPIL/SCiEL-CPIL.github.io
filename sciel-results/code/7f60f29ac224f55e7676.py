"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Uses a Chebyshev polynomial basis of degree 9 with smooth nonlinear
    parameterization via tanh to create a well-conditioned optimization
    landscape for BFGS. The tanh transform maps the raw params to a bounded
    range, preventing divergence and helping the optimizer find the global
    least-squares solution.
    
    The model: prediction = basis_std @ (scale * tanh(params))
    where basis_std is the standardized Chebyshev basis and scale controls
    the effective coefficient magnitude.
    """
    x_min = 0.1
    x_max = 6.0
    t = (2.0 * x - (x_max + x_min)) / (x_max - x_min)
    t = np.clip(t, -1.0, 1.0)
    
    n_points = x.shape[0]
    
    # Compute Chebyshev polynomials T_0 to T_9
    basis = np.empty((n_points, 10), dtype=np.float64)
    basis[:, 0] = 1.0
    basis[:, 1] = t
    
    for k in range(2, 10):
        basis[:, k] = 2.0 * t * basis[:, k-1] - basis[:, k-2]
        
    # Standardize each basis column using precomputed moments for uniform t in [-1, 1].
    # For T_k(t) with t ~ Uniform[-1, 1]:
    #   mean = 0 for all k (by symmetry)
    #   var(T_0) = 1
    #   var(T_k) = 0.5 for k >= 1
    std = np.ones(10, dtype=np.float64)
    std[1:] = np.sqrt(0.5)
    
    # Standardize the basis matrix
    basis_std = basis / std
    
    # Linear parameterization: params directly scale the standardized basis
    # This allows BFGS to find the true least-squares solution without
    # the bounded distortion introduced by tanh
    return basis_std @ params
# EVOLVE-BLOCK-END
