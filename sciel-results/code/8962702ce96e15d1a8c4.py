import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: Global quadratic polynomial + 5 local Gaussian RBFs.
    
    Parameters:
    params[0]: intercept
    params[1]: coeff for x
    params[2]: coeff for y
    params[3]: coeff for x^2
    params[4]: coeff for y^2
    params[5..9]: coeffs for 5 Gaussian RBFs
    
    RBF Centers (fixed):
    1. (0, 0)
    2. (0.6, 0.6)
    3. (-0.6, 0.6)
    4. (0.6, -0.6)
    5. (-0.6, -0.6)
    
    Sigma = 0.5
    """
    # Global Polynomial Terms
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    
    x2 = x * x
    y2 = y * y
    
    pred = p0 + p1 * x + p2 * y + p3 * x2 + p4 * y2
    
    # RBF Terms
    # Centers
    cx = np.array([0.0, 0.6, -0.6, 0.6, -0.6])
    cy = np.array([0.0, 0.6, 0.6, -0.6, -0.6])
    
    sigma2 = 0.5 * 0.5 # sigma^2
    
    # Calculate squared distances for each center
    # x shape: (N,), cx shape: (5,) -> broadcast to (N, 5)
    # We can do this efficiently
    
    # dx_i = x - cx_i
    # dist2_i = (x - cx_i)^2 + (y - cy_i)^2
    
    # Vectorized computation for all 5 centers at once
    # x: (N, 1), cx: (1, 5)
    x_col = x[:, np.newaxis]
    y_col = y[:, np.newaxis]
    
    dx = x_col - cx[np.newaxis, :]
    dy = y_col - cy[np.newaxis, :]
    
    dist2 = dx**2 + dy**2
    
    # Gaussian kernel: exp(-dist2 / (2 * sigma^2))
    # sigma^2 = 0.25, so 2*sigma^2 = 0.5
    rbf_values = np.exp(-dist2 / 0.5)
    
    # rbf_values shape: (N, 5)
    # params[5:] shape: (5,)
    
    # Sum the weighted RBFs
    rbf_contrib = np.sum(rbf_values * params[5:], axis=1)
    
    pred += rbf_contrib
    
    return pred
