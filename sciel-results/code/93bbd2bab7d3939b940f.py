"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF-based nonlinear oscillator acceleration.
    Uses Gaussian radial basis functions in the (x, v) input space.
    params[0:6] = centers (cx1, cy1, cx2, cy2, cx3, cy3)
    params[6:9] = widths (w1, w2, w3)
    params[9]   = global weight multiplier / bias
    """
    p = params
    # Extract parameters
    cx1, cy1, cx2, cy2, cx3, cy3 = p[0], p[1], p[2], p[3], p[4], p[5]
    w1, w2, w3 = p[6], p[7], p[8]
    bias = p[9]
    
    # Ensure widths are positive to avoid numerical issues
    w1 = np.abs(w1) + 1e-6
    w2 = np.abs(w2) + 1e-6
    w3 = np.abs(w3) + 1e-6
    
    # Compute squared distances to each center
    dx1 = x - cx1
    dy1 = v - cy1
    dist1 = dx1 * dx1 + dy1 * dy1
    
    dx2 = x - cx2
    dy2 = v - cy2
    dist2 = dx2 * dx2 + dy2 * dy2
    
    dx3 = x - cx3
    dy3 = v - cy3
    dist3 = dx3 * dx3 + dy3 * dy3
    
    # Gaussian RBFs
    rbf1 = np.exp(-dist1 / (2.0 * w1 * w1))
    rbf2 = np.exp(-dist2 / (2.0 * w2 * w2))
    rbf3 = np.exp(-dist3 / (2.0 * w3 * w3))
    
    # Combine with simple weighting: first 3 RBFs get equal weight scaled by bias
    # We use the bias parameter as a global scale, and distribute weights implicitly
    # Actually, with only 10 params, we need to be careful.
    # Let's use: output = bias * (rbf1 + rbf2 + rbf3) / 3
    # But this gives no individual weight control.
    # Better: use the centers' x-coordinates as implicit weights? No, that's weird.
    # Alternative: use a fixed structure where the 3 RBFs are summed, and bias scales it.
    # But we have 3 widths and 6 center coords and 1 bias = 10 params.
    # Let's just sum them and scale by bias.
    
    return bias * (rbf1 + rbf2 + rbf3) / 3.0
# EVOLVE-BLOCK-END
