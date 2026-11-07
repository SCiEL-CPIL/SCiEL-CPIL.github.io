"""Symbolic regression: fit a hybrid Chebyshev + Gaussian RBF basis via BFGS."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_TIME_BUDGET = 135.0  # safety margin before 150s kill

# Fixed hyperparameters for the hybrid basis
_N_CHEB = 5       # number of Chebyshev terms
_N_GAUSS = 5      # number of Gaussian RBF centers
_GAUSS_WIDTH = 0.6  # width parameter for Gaussians (in original x-scale)
# Centers spread across [-2, 2]
_GAUSS_CENTERS = np.array([-1.6, -0.8, 0.0, 0.8, 1.6])


def _remaining_time() -> float:
    """Return remaining seconds before safety deadline."""
    return _TIME_BUDGET - (time.time() - _START_TIME)


def _hybrid_basis(x, n_cheb=_N_CHEB, n_gauss=_N_GAUSS):
    """
    Compute a hybrid basis matrix:
      - First n_cheb columns: Chebyshev polynomials T_0..T_{n_cheb-1} on x/2
      - Next n_gauss columns: exp(-((x - c_i)^2) / (2 * sigma^2)) for each center c_i
    Returns a matrix of shape (len(x), n_cheb + n_gauss).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n_total = n_cheb + n_gauss
    basis = np.zeros((len(x), n_total))
    
    # --- Chebyshev part ---
    t = x / 2.0  # map to [-1, 1]
    basis[:, 0] = 1.0
    if n_cheb > 1:
        basis[:, 1] = t
    for k in range(2, n_cheb):
        basis[:, k] = 2.0 * t * basis[:, k - 1] - basis[:, k - 2]
    
    # --- Gaussian RBF part ---
    sigma2 = 2.0 * _GAUSS_WIDTH ** 2
    for i in range(n_gauss):
        c = _GAUSS_CENTERS[i]
        basis[:, n_cheb + i] = np.exp(-((x - c) ** 2) / sigma2)
    
    return basis


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Evaluate the hybrid model:
      sum_{i=0}^{n_cheb-1} params[i] * T_i(x/2) + sum_{j=0}^{n_gauss-1} params[n_cheb+j] * RBF_j(x)
    Linear in params, so BFGS converges well.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()
    
    # Ensure params has exactly 10 elements
    if len(params) < 10:
        params = np.concatenate([params, np.zeros(10 - len(params))])
    else:
        params = params[:10]
        
    # Compute the hybrid basis matrix
    basis = _hybrid_basis(x)
    
    # Compute predictions
    return basis @ params
# EVOLVE-BLOCK-END
