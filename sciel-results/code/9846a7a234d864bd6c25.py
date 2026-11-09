"""Rewritten program for symbolic regression with Gaussian RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear model using Gaussian Radial Basis Functions (RBFs).
    
    The input x is in [-2, 2]. We use 10 Gaussian kernels centered at
    distinct points within [-2, 2], with a fixed bandwidth sigma.
    
    The model is:
        f(x) = sum_{k=0}^{9} params[k] * exp(-((x - c_k)^2) / (2 * sigma^2))
    
    This is linear in params, so BFGS converges to the global minimum.
    The Gaussian basis provides smooth, highly flexible non-polynomial
    approximation without the Runge's phenomenon of high-degree polynomials.
    """
    # Centers spread evenly across [-2, 2]
    centers = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 0.0])
    
    # Bandwidth: wide enough to capture global structure but narrow enough
    # to capture local features. sigma ~ 0.5 gives good coverage.
    sigma = 0.5
    sigma2 = 2.0 * sigma * sigma
    
    # Compute all Gaussian basis values: shape (n_points, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    basis = np.exp(-(diff ** 2) / sigma2)
    
    # Linear combination: basis @ params
    return basis @ params
# EVOLVE-BLOCK-END
