"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard (mostly for compliance, actual computation is very fast)
    # The evaluator handles the main loop, but we ensure we don't hang.
    # In this specific task, 'equation' is just a forward pass.
    
    # Parameters for Kernel 1
    c1x, c1y, c1z = params[0], params[1], params[2]
    # Use exp to ensure positive width. Scale by 0.5 to match input range [0.5, 2]
    # If p=1, sigma = 0.5 * e ~ 1.35
    sigma1 = 0.5 * np.exp(params[3])
    w1 = params[4]
    
    # Parameters for Kernel 2
    c2x, c2y, c2z = params[5], params[6], params[7]
    sigma2 = 0.5 * np.exp(params[8])
    w2 = params[9]
    
    # Calculate distances
    dx1 = x - c1x
    dy1 = y - c1y
    dz1 = z - c1z
    dist1_sq = dx1**2 + dy1**2 + dz1**2
    
    dx2 = x - c2x
    dy2 = y - c2y
    dz2 = z - c2z
    dist2_sq = dx2**2 + dy2**2 + dz2**2
    
    # Calculate RBF values
    # Clip the exponent to avoid underflow/overflow issues if sigma is tiny
    # exp(-1000) is 0, which is fine. exp(1000) is inf, but dist^2 >= 0 so exponent <= 0.
    term1 = w1 * np.exp(-dist1_sq / (2.0 * sigma1**2))
    term2 = w2 * np.exp(-dist2_sq / (2.0 * sigma2**2))
    
    return term1 + term2
# EVOLVE-BLOCK-END
