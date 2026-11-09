import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF-based symbolic regression model.
    
    Instead of global trigonometric/polynomial basis, this uses 3 local 
    Gaussian RBFs. The 10 parameters define:
    - 3 centers (cx1, cy1), (cx2, cy2), (cx3, cy3)
    - 3 weights (w1, w2, w3)
    - 1 shared width (sigma)
    - 1 constant offset (c)
    
    This allows the model to capture local structures and non-linearities 
    that global basis functions might miss, provided the target function 
    has localized features or can be approximated by local Gaussians.
    """
    # Unpack parameters
    cx1, cy1 = params[0], params[1]
    cx2, cy2 = params[2], params[3]
    cx3, cy3 = params[4], params[5]
    w1, w2, w3 = params[6], params[7], params[8]
    sigma = params[9]
    
    # Ensure sigma is positive and not too small to avoid numerical issues
    # We use softplus-like transformation or just clip to ensure stability
    # Since BFGS can push sigma to negative, we use abs(sigma) + epsilon
    sigma_eff = np.abs(sigma) + 1e-6
    
    # Compute squared distances to centers
    # d1^2 = (x - cx1)^2 + (y - cy1)^2
    dx1 = x - cx1
    dy1 = y - cy1
    d1_sq = dx1**2 + dy1**2
    
    dx2 = x - cx2
    dy2 = y - cy2
    d2_sq = dx2**2 + dy2**2
    
    dx3 = x - cx3
    dy3 = y - cy3
    d3_sq = dx3**2 + dy3**2
    
    # Compute RBF values: exp(-d^2 / (2 * sigma^2))
    # Using 2*sigma^2 in denominator is standard Gaussian RBF
    inv_2_sigma_sq = 1.0 / (2.0 * sigma_eff**2)
    
    rbf1 = np.exp(-d1_sq * inv_2_sigma_sq)
    rbf2 = np.exp(-d2_sq * inv_2_sigma_sq)
    rbf3 = np.exp(-d3_sq * inv_2_sigma_sq)
    
    # Linear combination + constant
    val = w1 * rbf1 + w2 * rbf2 + w3 * rbf3
    
    # We don't have a parameter for constant offset explicitly if we use all 10 params.
    # Let's re-allocate: 
    # p[0:6] centers (3 RBFs * 2 coords)
    # p[6:9] weights (3 RBFs)
    # p[9] sigma
    # We are missing a constant term. 
    # To capture the mean, one of the RBFs could act as a broad constant if sigma is large and center is 0.
    # Alternatively, we can add a small linear term if we had more params, but we are stuck with 10.
    # Let's stick to 3 RBFs + shared sigma. The mean can be captured by a very wide RBF.
    
    return val
