"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of Gaussian Radial Basis Functions.
    This approach uses a fixed set of basis functions (Gaussians) centered
    across the input domain. The parameters `params` correspond to the weights
    of these basis functions. This transforms the problem into a linear
    least-squares optimization, which is more stable for BFGS than fitting
    high-degree polynomial coefficients directly.
    """
    # Define fixed centers for the RBFs across the range [-3, 3]
    # 10 parameters -> 10 RBFs
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Define a fixed width (standard deviation) for the Gaussians.
    # A width of 1.0 provides a good balance between smoothness and flexibility
    # for functions defined on [-3, 3].
    sigma = 1.0
    sigma2 = 2.0 * sigma * sigma
    
    # Compute the basis matrix: shape (N_points, 10)
    # Each column i is exp(-(x - centers[i])^2 / (2 * sigma^2))
    # We use broadcasting: x is (N, 1), centers is (1, 10)
    x_col = x.reshape(-1, 1)
    diff = x_col - centers.reshape(1, -1)
    basis = np.exp(-diff**2 / sigma2)
    
    # Linear combination of basis functions
    y = basis @ params
    
    return y
# EVOLVE-BLOCK-END
