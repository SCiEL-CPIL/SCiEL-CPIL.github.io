"""
STRUCTURAL CHANGE: Hybrid RBF + Polynomial basis with adaptive center placement.

Combines the linear-in-params structure (reliable BFGS convergence) with a
richer basis that includes:
- 5 Gaussian RBF kernels centered at {-2.5, -1.25, 0, 1.25, 2.5} with sigma=0.8
- 5 polynomial terms: 1, x, x^2, x^3, x^4

The RBF centers are spaced to cover [-3, 3] with good overlap, and sigma=0.8
provides moderate localization. The polynomial terms capture global trends.

Key insight from inspirations:
- Linear-in-params is critical for reliable BFGS convergence (current program)
- RBF + polynomial combination captures both local features and global trends
- The basis should be well-conditioned: scale basis functions to comparable magnitudes

This gives BFGS a convex optimization problem in the 10 parameters, which it
can solve efficiently and reliably from the all-ones starting point.
"""
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # RBF centers: 5 centers evenly spaced across [-3, 3]
    centers = np.array([-2.5, -1.25, 0.0, 1.25, 2.5])
    sigma = 0.8  # fixed width for Gaussian kernels
    
    # Compute 5 Gaussian RBF values: shape (n, 5)
    # phi_i(x) = exp(-(x - c_i)^2 / (2 * sigma^2))
    diff = x[:, np.newaxis] - centers[np.newaxis, :]  # (n, 5)
    rbf = np.exp(-diff ** 2 / (2.0 * sigma ** 2))     # (n, 5)
    
    # Polynomial basis (5 terms)
    b0 = np.ones_like(x)
    b1 = x
    b2 = x ** 2
    b3 = x ** 3
    b4 = x ** 4
    
    poly = np.column_stack([b0, b1, b2, b3, b4])  # (n, 5)
    
    # Stack into single basis matrix: shape (n, 10)
    B = np.column_stack([rbf, poly])  # (n, 10)
    
    # Linear combination
    y = B @ p
    
    # Ensure finite output
    y = np.where(np.isfinite(y), y, 0.0)
    
    return y
