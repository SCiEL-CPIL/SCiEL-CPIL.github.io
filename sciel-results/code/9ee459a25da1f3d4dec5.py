"""RBF network for nonlinear oscillator acceleration."""
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using 2 Gaussian RBFs + linear baseline.
    
    Parameters:
        x: position array
        v: velocity array
        params: array of 10 values
            p[0] = RBF1 center_x (scaled)
            p[1] = RBF1 center_v (scaled)
            p[2] = RBF1 log-width (softplus for positivity)
            p[3] = RBF1 weight
            p[4] = RBF2 center_x (scaled)
            p[5] = RBF2 center_v (scaled)
            p[6] = RBF2 log-width (softplus for positivity)
            p[7] = RBF2 weight
            p[8] = linear x coefficient
            p[9] = linear v coefficient
    """
    # Scale inputs to roughly [-1, 1] for numerical stability
    xs = x / 0.6
    vs = v / 0.3
    
    # Extract parameters
    cx1 = params[0]
    cv1 = params[1]
    w1_raw = params[2]
    a1 = params[3]
    cx2 = params[4]
    cv2 = params[5]
    w2_raw = params[6]
    a2 = params[7]
    lx = params[8]
    lv = params[9]
    
    # Use softplus for widths to ensure positive sigma_sq
    # Clip raw values for numerical stability in exp
    w1_clipped = np.clip(w1_raw, -50, 50)
    w2_clipped = np.clip(w2_raw, -50, 50)
    sigma1_sq = np.log1p(np.exp(w1_clipped))
    sigma2_sq = np.log1p(np.exp(w2_clipped))
    
    # Ensure minimum width to avoid division issues
    sigma1_sq = np.maximum(sigma1_sq, 1e-8)
    sigma2_sq = np.maximum(sigma2_sq, 1e-8)
    
    # Compute squared distances in scaled coordinates
    dx1 = xs - cx1
    dv1 = vs - cv1
    r1_sq = dx1 * dx1 + dv1 * dv1
    
    dx2 = xs - cx2
    dv2 = vs - cv2
    r2_sq = dx2 * dx2 + dv2 * dv2
    
    # Gaussian RBFs: exp(-r^2 / (2*sigma^2))
    rbf1 = a1 * np.exp(-r1_sq / (2.0 * sigma1_sq))
    rbf2 = a2 * np.exp(-r2_sq / (2.0 * sigma2_sq))
    
    # Linear baseline in scaled coords
    linear = lx * xs + lv * vs
    
    return rbf1 + rbf2 + linear
# EVOLVE-BLOCK-END
