"""Symbolic regression with 1 input using a hybrid Chebyshev + Gaussian kernel basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Combine a rich set of well-conditioned basis functions with 10 parameters.
    
    The basis includes:
    - Low-degree Chebyshev polynomials (T_0 to T_2) for global smooth structure
    - 7 Gaussian kernels centered at evenly spaced points for local adaptation
    
    This hybrid approach allows the model to capture both the overall trend
    (via Chebyshev) and local variations (via Gaussian kernels).
    
    The 10 parameters are:
    - params[0:3]: weights for T_0, T_1, T_2 (Chebyshev polynomials)
    - params[3:10]: weights for 7 Gaussian kernels centered at evenly spaced points
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    if len(x) == 0:
        return np.empty(0, dtype=np.float64)
    
    # Map x from [0.1, 6.0] to u in [-1, 1]
    x_min = 0.1
    x_max = 6.0
    center = (x_min + x_max) / 2.0
    scale = (x_max - x_min) / 2.0
    
    x_safe = np.clip(x, x_min, x_max)
    u = (x_safe - center) / scale
    
    n = len(x)
    
    # Build basis matrix with 10 columns
    basis = np.empty((n, 10), dtype=np.float64)
    
    # Columns 0-2: Chebyshev polynomials T_0, T_1, T_2
    basis[:, 0] = 1.0
    basis[:, 1] = u
    basis[:, 2] = 2.0 * u * u - 1.0
    
    # Columns 3-9: Gaussian kernels centered at 7 evenly spaced points in [-1, 1]
    # Centers: -1, -0.667, -0.333, 0, 0.333, 0.667, 1
    kernel_centers = np.linspace(-1.0, 1.0, 7)
    # Bandwidth: use a fixed bandwidth that gives reasonable localization
    # sigma = 0.25 gives kernels with ~80% mass within +/-0.5 units
    sigma = 0.25
    
    for i, c in enumerate(kernel_centers):
        # Gaussian kernel: exp(-((u - c)^2) / (2 * sigma^2))
        basis[:, 3 + i] = np.exp(-((u - c) ** 2) / (2.0 * sigma * sigma))
    
    # Linear combination of basis functions with parameters
    return basis @ params
# EVOLVE-BLOCK-END
