"""Symbolic regression with 1 input: flexible local basis approximation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Use a mixture of Gaussian RBFs + global polynomial terms to approximate
    an unknown function on x in [-2, 2].
    
    params[0:6] -> coefficients for 6 Gaussian RBFs (centers spread over [-2,2])
    params[6]   -> coefficient for constant term
    params[7]   -> coefficient for x
    params[8]   -> coefficient for x^2
    params[9]   -> coefficient for x^3
    
    The RBF widths are fixed to provide local flexibility.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Fixed centers for RBFs, spread across [-2, 2]
    centers = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 0.5])
    
    # Fixed width (sigma) for each RBF
    sigma = 0.8
    sigma2 = 2.0 * sigma * sigma
    
    # Compute RBF features: exp(-(x - c)^2 / (2*sigma^2))
    # Shape: (n_points, 6)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_features = np.exp(-diff**2 / sigma2)
    
    # Global polynomial features
    poly_features = np.column_stack([
        np.ones_like(x),
        x,
        x**2,
        x**3
    ])
    
    # Combine: params[0:6] for RBFs, params[6:10] for polynomials
    rbf_coeffs = params[0:6]
    poly_coeffs = params[6:10]
    
    prediction = rbf_features @ rbf_coeffs + poly_features @ poly_coeffs
    
    return prediction
# EVOLVE-BLOCK-END
