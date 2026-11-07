"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Improved Gaussian RBF Network with 2 kernels, shared width, and a bias term.
    
    Parameters layout (10 total):
      - params[0]:   bias (b)
      - params[1:4]: center of kernel 1 (cx1, cy1, cz1)
      - params[4]:   shared width (sigma)
      - params[5]:   weight of kernel 1 (w1)
      - params[6:9]: center of kernel 2 (cx2, cy2, cz2)
      - params[9]:   weight of kernel 2 (w2)
    
    Output: b + w1 * exp(-||[x,y,z] - c1||^2 / (2*sigma^2)) 
          + w2 * exp(-||[x,y,z] - c2||^2 / (2*sigma^2))
    """
    # Extract parameters
    b = params[0]
    cx1, cy1, cz1 = params[1], params[2], params[3]
    sigma = params[4]
    w1 = params[5]
    cx2, cy2, cz2 = params[6], params[7], params[8]
    w2 = params[9]
    
    # Compute squared distances to each center
    dx1 = x - cx1
    dy1 = y - cy1
    dz1 = z - cz1
    dist1_sq = dx1 * dx1 + dy1 * dy1 + dz1 * dz1
    
    dx2 = x - cx2
    dy2 = y - cy2
    dz2 = z - cz2
    dist2_sq = dx2 * dx2 + dy2 * dy2 + dz2 * dz2
    
    # Clamp sigma to avoid division by zero or extremely small values
    # A minimum sigma ensures the kernel isn't a delta function
    eps = 1e-6
    s = np.maximum(np.abs(sigma), eps)
    s_sq = s * s
    
    # Compute Gaussian kernels
    # The exponent is always <= 0, so exp is stable (range (0, 1])
    k1 = np.exp(-dist1_sq / (2.0 * s_sq))
    k2 = np.exp(-dist2_sq / (2.0 * s_sq))
    
    # Weighted sum with bias
    result = b + w1 * k1 + w2 * k2
    
    return result
# EVOLVE-BLOCK-END
