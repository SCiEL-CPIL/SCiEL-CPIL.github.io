"""Physics-informed RBF representation for nonlinear oscillator."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a radial basis function representation.
    
    The 10 parameters are interpreted as:
    - params[0:2]: center (cx, cv) in (x, v) space
    - params[2:4]: second center
    - params[4:6]: third center
    - params[6:8]: fourth center
    - params[8]: global width (shared for numerical stability)
    - params[9]: bias/offset
    
    Each RBF is a Gaussian: exp(-((x-cx)^2 + (v-cv)^2) / (2*width^2))
    The output is a weighted sum of these Gaussians plus a bias.
    """
    # Extract centers and width
    cx1, cv1 = params[0], params[1]
    cx2, cv2 = params[2], params[3]
    cx3, cv3 = params[4], params[5]
    cx4, cv4 = params[6], params[7]
    width = np.abs(params[8]) + 1e-6  # Ensure positive width
    bias = params[9]
    
    # Compute squared distances
    dx1 = x - cx1
    dv1 = v - cv1
    dx2 = x - cx2
    dv2 = v - cv2
    dx3 = x - cx3
    dv3 = v - cv3
    dx4 = x - cx4
    dv4 = v - cv4
    
    dist1 = dx1**2 + dv1**2
    dist2 = dx2**2 + dv2**2
    dist3 = dx3**2 + dv3**2
    dist4 = dx4**2 + dv4**2
    
    # Gaussian RBFs
    scale = 1.0 / (2.0 * width**2)
    rbf1 = np.exp(-scale * dist1)
    rbf2 = np.exp(-scale * dist2)
    rbf3 = np.exp(-scale * dist3)
    rbf4 = np.exp(-scale * dist4)
    
    # Linear combination with fixed weights (all 1.0 for simplicity)
    # The centers and width are what get optimized
    result = rbf1 + rbf2 + rbf3 + rbf4 + bias
    
    return result
# EVOLVE-BLOCK-END
