"""
STRUCTURAL CHANGE: Radial Basis Function (RBF) architecture with polynomial trend.

Uses 5 fixed Gaussian RBF centers at {-3, -1.5, 0, 1.5, 3} with fixed width,
plus 5 low-order polynomial terms (1, x, x^2, x^3, x^4).

The RBF centers provide localized flexibility to capture non-periodic smooth
structure, while the polynomial terms capture global trends. All 10 parameters
are linear in the output, so BFGS can fit them efficiently (the problem is
convex in params for fixed architecture).

Parameters (10 total):
- params[0:5]  = coefficients for polynomial terms [1, x, x^2, x^3, x^4]
- params[5:10] = coefficients for RBF terms centered at [-3, -1.5, 0, 1.5, 3]

Fixed RBF width: sigma = 1.0 (gives reasonable local coverage over [-3,3])
"""
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Polynomial basis: 1, x, x^2, x^3, x^4
    poly_basis = np.column_stack([
        np.ones_like(x),
        x,
        x**2,
        x**3,
        x**4
    ])  # shape (n, 5)
    
    # RBF basis: Gaussian kernels with fixed centers and width
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    sigma = 1.0  # fixed width
    
    # Compute RBF values: exp(-((x - c_i)^2) / (2 * sigma^2))
    # x has shape (n,), centers has shape (5,)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]  # shape (n, 5)
    rbf_basis = np.exp(-diff**2 / (2.0 * sigma**2))  # shape (n, 5)
    
    # Combine: y = poly_basis @ p[0:5] + rbf_basis @ p[5:10]
    y = poly_basis @ p[0:5] + rbf_basis @ p[5:10]
    
    # Ensure finite output
    y = np.where(np.isfinite(y), y, 0.0)
    
    return y
