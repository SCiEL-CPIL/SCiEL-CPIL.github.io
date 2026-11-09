"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression candidate using Radial Basis Functions (RBF).
    
    This approach uses a local approximation strategy, distinct from the global
    polynomial/trigonometric basis used in the previous version.
    It models the function as a weighted sum of Gaussian bumps centered at fixed
    locations in the input space.
    
    params is an array of length 10. We use all 10 parameters as weights for:
    10 Gaussian RBFs with fixed centers covering the domain [-1, 1] x [-1, 1].
    """
    # Ensure x and y are 1D arrays
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    
    # Fixed centers for the RBFs
    # We select 10 strategic points: Center, 4 edge midpoints, 4 corners, and one additional point
    centers = np.array([
        [0.0, 0.0],       # 1: Center
        [1.0, 0.0],       # 2: Right
        [-1.0, 0.0],      # 3: Left
        [0.0, 1.0],       # 4: Top
        [0.0, -1.0],      # 5: Bottom
        [1.0, 1.0],       # 6: Top-Right
        [-1.0, 1.0],      # 7: Top-Left
        [1.0, -1.0],      # 8: Bottom-Right
        [-1.0, -1.0],     # 9: Bottom-Left
        [0.5, 0.5]        # 10: Additional point to improve resolution in quadrant
    ])
    
    # Width parameter for the Gaussian kernels
    # A value of 0.6 provides a good balance between locality and coverage for this domain
    sigma = 0.6
    sigma_sq = 2.0 * sigma**2
    
    # Compute the prediction
    pred = np.zeros_like(x)
    
    for i in range(10):
        cx, cy = centers[i]
        # Calculate squared Euclidean distance from each point (x, y) to the center (cx, cy)
        dist_sq = (x - cx)**2 + (y - cy)**2
        # Gaussian RBF value
        rbf_val = np.exp(-dist_sq / sigma_sq)
        # Accumulate weighted contribution
        pred += params[i] * rbf_val
        
    return pred
# EVOLVE-BLOCK-END
