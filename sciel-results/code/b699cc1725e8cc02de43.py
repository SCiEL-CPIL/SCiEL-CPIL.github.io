import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a hybrid model:
    Global linear trend + 4 adaptive Gaussian kernels.
    
    Parameters:
    p0, p1: Linear coefficients (intercept, slope)
    p2, p3: Center, WidthParam for Gaussian 1
    p4, p5: Center, WidthParam for Gaussian 2
    p6, p7: Center, WidthParam for Gaussian 3
    p8, p9: Center, WidthParam for Gaussian 4
    
    Widths are squared to ensure positivity: w_i = p_{2i+3}^2 + 1e-6
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0:
        x = x.reshape(1)
        
    # Extract parameters
    p0, p1, c1, w1_raw, c2, w2_raw, c3, w3_raw, c4, w4_raw = params
    
    # Calculate global linear part
    y = p0 + p1 * x
    
    # Calculate Gaussian parts
    # Use squared width to ensure positivity and smooth gradients near 0
    eps = 1e-8
    w1 = w1_raw**2 + eps
    w2 = w2_raw**2 + eps
    w3 = w3_raw**2 + eps
    w4 = w4_raw**2 + eps
    
    # Gaussian 1
    y += np.exp(-((x - c1)**2) / (2.0 * w1))
    # Gaussian 2
    y += np.exp(-((x - c2)**2) / (2.0 * w2))
    # Gaussian 3
    y += np.exp(-((x - c3)**2) / (2.0 * w3))
    # Gaussian 4
    y += np.exp(-((x - c4)**2) / (2.0 * w4))
    
    return y
