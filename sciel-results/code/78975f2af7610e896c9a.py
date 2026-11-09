"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the unknown function using localized Gaussian RBFs with varying widths
    plus a global linear trend. This allows capturing local features, sharp transitions,
    and non-smooth behavior that global polynomial/trig bases miss.
    
    Basis functions (10 total):
    0: 1 (constant)
    1: x (linear trend)
    2: Gaussian RBF centered at -2.5, width 1.0
    3: Gaussian RBF centered at -1.5, width 0.7
    4: Gaussian RBF centered at -0.5, width 0.5
    5: Gaussian RBF centered at 0.5, width 0.5
    6: Gaussian RBF centered at 1.5, width 0.7
    7: Gaussian RBF centered at 2.5, width 1.0
    8: Wide Gaussian RBF centered at 0, width 2.0 (global smooth component)
    9: Very wide Gaussian RBF centered at 0, width 3.5 (ultra-global)
    """
    x = np.asarray(x, dtype=np.float64)
    original_shape = x.shape
    x_flat = x.flatten()
    
    if x_flat.size == 0:
        return np.array([]).reshape(original_shape)
    
    # Global components
    b0 = np.ones_like(x_flat)
    b1 = x_flat
    
    # Localized Gaussian RBFs with distinct centers and widths
    # Centers spread across [-3, 3], widths vary to capture different scales
    centers = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 0.0, 0.0])
    widths = np.array([1.0, 0.7, 0.5, 0.5, 0.7, 1.0, 2.0, 3.5])
    
    # Precompute 2 * width^2 for efficiency
    two_width_sq = 2.0 * widths * widths
    
    # Compute all RBFs vectorized
    # diff shape: (n_points, 8)
    diff = x_flat[:, np.newaxis] - centers[np.newaxis, :]
    rbf_matrix = np.exp(-diff * diff / two_width_sq[np.newaxis, :])
    
    # Extract individual RBFs
    b2 = rbf_matrix[:, 0]
    b3 = rbf_matrix[:, 1]
    b4 = rbf_matrix[:, 2]
    b5 = rbf_matrix[:, 3]
    b6 = rbf_matrix[:, 4]
    b7 = rbf_matrix[:, 5]
    b8 = rbf_matrix[:, 6]
    b9 = rbf_matrix[:, 7]
    
    # Linear combination
    y = (params[0] * b0 +
         params[1] * b1 +
         params[2] * b2 +
         params[3] * b3 +
         params[4] * b4 +
         params[5] * b5 +
         params[6] * b6 +
         params[7] * b7 +
         params[8] * b8 +
         params[9] * b9)
    
    # Ensure finite values
    y = np.where(np.isfinite(y), y, 0.0)
    
    return y.reshape(original_shape)
# EVOLVE-BLOCK-END
