"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Radial Basis Function (RBF) model with 5 Gaussian centers and 5 polynomial terms.
    
    params layout:
      params[0:5]   -> coefficients for polynomial terms: [1, x, x^2, x^3, x^4]
      params[5:10]  -> coefficients for RBF Gaussian kernels at centers {-3, -1.5, 0, 1.5, 3}
    
    The Gaussian width is fixed at sigma = 1.0 to ensure smooth, localized bumps
    that BFGS can fit independently without overfitting or numerical instability.
    """
    x = np.asarray(x, dtype=float)
    
    # Polynomial basis: 1, x, x^2, x^3, x^4
    poly_terms = np.stack([
        np.ones_like(x),
        x,
        x ** 2,
        x ** 3,
        x ** 4
    ], axis=-1)  # shape: (N, 5)
    
    # RBF Gaussian kernels with fixed centers and sigma=1.0
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    sigma = 1.0
    
    # Compute Gaussian kernels: shape (N, 5)
    # diff shape: (N, 5)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_terms = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Combine polynomial and RBF terms: shape (N, 10)
    basis = np.hstack([poly_terms, rbf_terms])
    
    # Prediction: dot product with params
    prediction = basis @ params
    
    return prediction
# EVOLVE-BLOCK-END
