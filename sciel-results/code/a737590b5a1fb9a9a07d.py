"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a Gaussian RBF basis.
    
    Uses 10 parameters as 5 RBF centers in (x, v) space.
    Each RBF: exp(-((x-cx)^2 + (v-cv)^2) / (2*sigma^2))
    with sigma fixed to cover the input domain.
    Output = sum of RBF values (weights are implicitly 1, centers are fitted).
    
    This provides a smooth, localized nonlinear basis that avoids
    ill-conditioning of high-order polynomials while capturing
    complex oscillator dynamics.
    """
    # Input ranges for normalization
    t_min, t_max = 30.0, 50.0
    x_min, x_max = -0.2056, 0.1749
    v_min, v_max = -0.3391, 0.3624
    
    # Normalize inputs to [0, 1] range
    t_norm = (t - t_min) / (t_max - t_min)
    x_norm = (x - x_min) / (x_max - x_min)
    v_norm = (v - v_min) / (v_max - v_min)
    
    # 5 RBF centers in (x, v) space, using 10 params
    # Centers are in normalized [0,1] space
    cx1, cv1, cx2, cv2, cx3, cv3, cx4, cv4, cx5, cv5 = params
    
    # Fixed width: covers ~40% of the domain in each dimension
    sigma = 0.2
    
    # Compute squared distances for each RBF
    dx1 = x_norm - cx1
    dv1 = v_norm - cv1
    dx2 = x_norm - cx2
    dv2 = v_norm - cv2
    dx3 = x_norm - cx3
    dv3 = v_norm - cv3
    dx4 = x_norm - cx4
    dv4 = v_norm - cv4
    dx5 = x_norm - cx5
    dv5 = v_norm - cv5
    
    # Gaussian RBF values
    rbf1 = np.exp(-(dx1**2 + dv1**2) / (2.0 * sigma**2))
    rbf2 = np.exp(-(dx2**2 + dv2**2) / (2.0 * sigma**2))
    rbf3 = np.exp(-(dx3**2 + dv3**2) / (2.0 * sigma**2))
    rbf4 = np.exp(-(dx4**2 + dv4**2) / (2.0 * sigma**2))
    rbf5 = np.exp(-(dx5**2 + dv5**2) / (2.0 * sigma**2))
    
    # Sum of RBFs as the acceleration prediction
    # Add a small constant offset to help with baseline
    return rbf1 + rbf2 + rbf3 + rbf4 + rbf5
# EVOLVE-BLOCK-END
