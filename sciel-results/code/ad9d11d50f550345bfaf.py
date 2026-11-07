"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of Gaussian RBFs with fixed centers and widths.
    
    The 10 parameters are local weights for 10 RBFs evenly spaced across [0.1, 6].
    This provides a well-conditioned, decoupled optimization landscape for BFGS.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Fixed centers evenly spaced across the domain [0.1, 6]
    # 10 centers
    x_min = 0.1
    x_max = 6.0
    centers = np.linspace(x_min, x_max, 10)
    
    # Fixed widths: choose sigma so that RBFs have reasonable overlap
    # Distance between adjacent centers
    spacing = (x_max - x_min) / (10 - 1)
    # Set width to be a fraction of the spacing for good local coverage
    # sigma = spacing / 2 gives good overlap
    sigma = spacing / 2.0
    
    # Compute RBF values: phi_i(x) = exp(-((x - c_i)^2) / (2 * sigma^2))
    # Shape: (n_points, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_values = np.exp(-(diff ** 2) / (2.0 * sigma ** 2))
    
    # Linear combination: prediction = sum_i params[i] * phi_i(x)
    prediction = rbf_values @ params
    
    return prediction
# EVOLVE-BLOCK-END
