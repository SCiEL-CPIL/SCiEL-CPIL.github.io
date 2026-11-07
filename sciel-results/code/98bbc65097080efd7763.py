# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression: recover an unknown function of 1 input.
    
    Uses a linear combination of 9 RBFs at fixed, evenly spaced centers.
    The first parameter controls the shared width (sigma) of all kernels.
    The remaining 9 parameters are coefficients for the 9 RBFs.
    
    param[0]: width parameter (sigma = exp(param[0]), always positive)
    param[1:10]: coefficients for 9 RBFs at centers -3, -2.25, ..., 3
    
    The RBF model is smooth and well-conditioned for reasonable sigma values,
    allowing BFGS to efficiently optimize both the width and coefficients.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Guard against non-finite inputs
    x = np.where(np.isfinite(x), x, 0.0)
    
    # Width parameter (always positive via exp)
    sigma = np.exp(params[0])
    
    # 9 evenly spaced centers in [-3, 3]
    centers = np.linspace(-3.0, 3.0, 9)
    
    # Build RBF matrix: N x 9
    # rbf_j(x) = exp(-((x - c_j)^2) / (2 * sigma^2))
    X = x[:, np.newaxis] - centers[np.newaxis, :]
    R = np.exp(-X**2 / (2.0 * sigma**2))
    
    # Coefficients
    coeffs = params[1:10]
    
    # Linear combination of RBFs
    result = R @ coeffs
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
