"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of Gaussian RBFs with fixed centers and widths.
    10 parameters act as independent local weights.
    
    Centers are evenly spaced across [0.1, 6.0].
    Widths are chosen to provide good local coverage.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # 10 RBF centers evenly spaced in [0.1, 6.0]
    # Use np.linspace to place them
    n_rbf = 10
    x_min, x_max = 0.1, 6.0
    centers = np.linspace(x_min + 0.3, x_max - 0.3, n_rbf)  # slight inset to avoid edge effects
    
    # Fixed widths: choose sigma so that adjacent centers have some overlap
    # Distance between adjacent centers:
    spacing = (x_max - x_min - 0.6) / (n_rbf - 1)
    # Sigma ~ spacing / 1.5 to have reasonable overlap
    sigma = spacing / 1.5
    
    # Compute RBF values: phi_i(x) = exp(-((x - c_i)^2) / (2 * sigma^2))
    # Shape: (n_points, n_rbf)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]  # (n_points, 10)
    rbf = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Linear combination: rbf @ params
    prediction = rbf @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(prediction)):
        # Fallback to a simple linear model if something goes wrong
        return params[0] * x + params[1]
    
    return prediction
# EVOLVE-BLOCK-END
