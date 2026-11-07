"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF model with 5 polynomial terms and 5 Gaussian RBF terms.
    
    Design:
    1. Normalize input x from [-3, 3] to z in [-1, 1].
    2. Use 5 polynomial terms: 1, z, z^2, z^3, z^4.
    3. Use 5 Gaussian RBF terms with centers at -1, -0.5, 0, 0.5, 1 (corresponding to -3, -1.5, 0, 1.5, 3 in x-space).
    4. Fixed width sigma = 0.5 in z-space.
    
    params layout:
      params[0:5]  -> coefficients for [1, z, z^2, z^3, z^4]
      params[5:10] -> coefficients for 5 Gaussian kernels
    """
    x = np.asarray(x, dtype=float)
    
    # Normalize input to [-1, 1]
    z = x / 3.0
    
    # Polynomial basis: Chebyshev polynomials T0 to T4
    # This improves numerical conditioning compared to monomials
    z2 = z ** 2
    z3 = z2 * z
    z4 = z2 * z2
    
    poly_terms = np.stack([
        np.ones_like(z),
        z,
        2 * z2 - 1,
        4 * z3 - 3 * z,
        8 * z4 - 8 * z2 + 1
    ], axis=-1)  # shape: (N, 5)
    
    # RBF Gaussian kernels
    # 5 centers corresponding to {-3, -1.5, 0, 1.5, 3} in x-space
    centers = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    # Reduced sigma to improve orthogonality between RBFs
    sigma = 0.3
    
    # Compute Gaussian kernels: shape (N, 5)
    diff = z[:, np.newaxis] - centers[np.newaxis, :]
    rbf_terms = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Combine polynomial and RBF terms: shape: (N, 10)
    basis = np.hstack([poly_terms, rbf_terms])
    
    # Prediction: dot product with params
    prediction = basis @ params
    
    return prediction
# EVOLVE-BLOCK-END
