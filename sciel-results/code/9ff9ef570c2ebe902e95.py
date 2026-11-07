"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Improved RBF model with normalized input and mixed polynomial/RBF basis.
    
    Design:
    1. Normalize input x from [-3, 3] to z in [-1, 1] for numerical stability.
    2. Use 3 low-order polynomial terms: 1, z, z^2.
    3. Use 7 Gaussian RBF terms with centers evenly spaced in [-1, 1].
    4. Fixed width sigma = 0.4 in z-space.
    
    params layout:
      params[0:3]  -> coefficients for [1, z, z^2]
      params[3:10] -> coefficients for 7 Gaussian kernels
    """
    x = np.asarray(x, dtype=float)
    
    # Normalize input to [-1, 1]
    z = x / 3.0
    
    # Polynomial basis: 1, z, z^2
    poly_terms = np.stack([
        np.ones_like(z),
        z,
        z ** 2
    ], axis=-1)  # shape: (N, 3)
    
    # RBF Gaussian kernels
    # 7 centers evenly spaced in [-1, 1]
    centers = np.linspace(-1.0, 1.0, 7)
    sigma = 0.4
    
    # Compute Gaussian kernels: shape (N, 7)
    diff = z[:, np.newaxis] - centers[np.newaxis, :]
    rbf_terms = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Combine polynomial and RBF terms: shape (N, 10)
    basis = np.hstack([poly_terms, rbf_terms])
    
    # Prediction: dot product with params
    prediction = basis @ params
    
    return prediction
# EVOLVE-BLOCK-END
