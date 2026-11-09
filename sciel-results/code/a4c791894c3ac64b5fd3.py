import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a flexible basis of polynomials and
    localized RBF-like features, all parameterized by the 10 fitted params.
    
    The basis is designed to be smooth and able to represent a wide variety of
    functions on [-3, 3]:
    - Constant term
    - Linear term
    - Quadratic term
    - Cubic term
    - Sine and cosine terms (for periodic-like behavior)
    - Several RBF-like Gaussian bumps at fixed centers
    
    The params are the coefficients for this fixed basis.
    """
    # Ensure x is 1D
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    
    # Fixed centers for RBF features
    centers = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    # Build the basis matrix: n x 10
    # Columns:
    # 0: constant
    # 1: x
    # 2: x^2
    # 3: x^3
    # 4: sin(x)
    # 5: cos(x)
    # 6-10: RBF at centers[0..4] with fixed width
    
    basis = np.empty((n, 10), dtype=np.float64)
    
    basis[:, 0] = 1.0
    basis[:, 1] = x
    basis[:, 2] = x * x
    basis[:, 3] = x * x * x
    basis[:, 4] = np.sin(x)
    basis[:, 5] = np.cos(x)
    
    # RBF features: exp(-(x - c)^2 / (2 * sigma^2))
    sigma = 0.8
    for i, c in enumerate(centers):
        diff = x - c
        basis[:, 6 + i] = np.exp(-(diff * diff) / (2.0 * sigma * sigma))
    
    # Compute prediction: basis @ params
    # params has shape (10,)
    y = basis @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(y)):
        # Fallback to zero if something went wrong
        return np.zeros(n)
    
    return y
