"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Strategy: Kernel Ridge Regression with Gaussian RBFs
#
# The model is a linear combination of Gaussian RBF kernels with fixed, evenly spaced centers.
# This allows the 10 parameters to act as adaptive weights for local interpolation.
# The basis functions are:
#   1. A constant term (for the global offset)
#   9. Gaussian RBFs centered at evenly spaced points in [-3, 3]
#
# This shifts the representation from global smoothness constraints (polynomials/trig) to local flexibility,
# enabling the external BFGS optimizer to fit the weights more effectively.
#
# To improve conditioning and convergence:
# 1. The centers are chosen to cover the domain [-3, 3] uniformly.
# 2. The width (sigma) of the RBFs is chosen to be moderate (e.g., 1.0) to balance local and global behavior.
# 3. The constant term is included to capture the mean of the function.

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Kernel Ridge Regression Model with Gaussian RBFs.
    
    Args:
        x: Input array of shape (N,).
        params: Array of 10 parameters.
            params[0] = Weight for the constant term
            params[1:10] = Weights for 9 Gaussian RBFs
            
    Returns:
        Prediction array of shape (N,).
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(params, dtype=float)
    
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)
        
    # 1. Constant term
    const_part = p[0]
    
    # 2. Gaussian RBF Part
    N_RBF = 9
    # Centers: Evenly spaced in [-3, 3]
    # np.linspace(-3, 3, 9) gives centers at -3, -2.25, -1.5, -0.75, 0, 0.75, 1.5, 2.25, 3
    centers = np.linspace(-3.0, 3.0, N_RBF)
    
    # Fixed width
    sigma = 1.0
    sigma_sq_2 = 2.0 * sigma**2
    
    # Compute RBF values
    # x shape: (N,)
    # centers shape: (9,)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_values = np.exp(-(diff**2) / sigma_sq_2)
    
    # rbf_values shape: (N, 9)
    # p[1:10] shape: (9,)
    rbf_part = rbf_values @ p[1:10]
    
    result = const_part + rbf_part
    
    return result

# EVOLVE-BLOCK-END
