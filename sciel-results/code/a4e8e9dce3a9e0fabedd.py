"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF kernel basis model with 10 adaptive weights.
    
    Uses 10 fixed RBF centers spread across the domain [0.1, 6].
    Each center has a Gaussian kernel with fixed bandwidth.
    params are transformed via exp() to ensure positive weights.
    
    This allows fitting arbitrary smooth functions via local interpolation.
    """
    x = np.asarray(x, dtype=np.float64)
    x_flat = x.flatten()
    
    # 10 RBF centers evenly spaced in [0.1, 6]
    centers = np.linspace(0.1, 6.0, 10)
    
    # Fixed bandwidth - tuned for domain scale
    # Domain width is ~5.9, so bandwidth ~1.0 gives good local coverage
    bandwidth = 1.0
    
    # Compute RBF basis: 10 kernels evaluated at x
    # Shape: (n_points, 10)
    x_expanded = x_flat[:, np.newaxis]  # (n, 1)
    centers_expanded = centers[np.newaxis, :]  # (1, 10)
    dist_sq = (x_expanded - centers_expanded) ** 2  # (n, 10)
    rbf = np.exp(-dist_sq / (2.0 * bandwidth ** 2))  # (n, 10)
    
    # Transform params to positive weights
    weights = np.exp(params)  # (10,)
    
    # Prediction: sum of weighted RBFs
    pred = rbf @ weights  # (n,)
    
    return pred.reshape(x.shape)
# EVOLVE-BLOCK-END
