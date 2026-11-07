"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Uses a Chebyshev polynomial basis of degree 9, with each basis column
    standardized (zero mean, unit variance) using fixed constants derived
    from the known input domain [0.1, 6].
    
    The standardization ensures the design matrix is well-conditioned,
    allowing BFGS to converge reliably to the global least-squares solution.
    Since the mapping from standardized coefficients to raw Chebyshev
    coefficients is linear and invertible, the model class is identical
    to the unstandardized version, but optimization is far more stable.
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
    # We use these exact values to standardize.
    
    # Precomputed standardization parameters:
    # std[k] = 1.0 for k=0, sqrt(0.5) for k>=1
    # mean[k] = 0.0 for all k
    
    # To avoid division by zero or instability, we scale columns:
    # standardized_col = (col - mean) / std = col / std  (since mean=0)
    
    std = np.ones(10, dtype=np.float64)
    std[1:] = np.sqrt(0.5)
    
    # Standardize the basis matrix
    basis_std = basis / std
    
    # The model is basis_std @ params_std, but the caller passes params in the
    # original (unstandardized) coefficient space? No -- the caller fits params
    # via BFGS minimizing MSE. The params are whatever the optimizer finds.
    # By standardizing the basis, we make the optimization landscape well-conditioned.
    # The final prediction is basis_std @ params, where params are the fitted
    # standardized coefficients.
    
    return basis_std @ params
# EVOLVE-BLOCK-END
