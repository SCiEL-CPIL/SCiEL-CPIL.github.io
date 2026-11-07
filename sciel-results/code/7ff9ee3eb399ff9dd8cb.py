import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a hybrid basis combining:
    - Low-order polynomial terms for global smooth structure
    - Gaussian RBF kernels for local non-polynomial features
    
    The 10 parameters define:
    - p[0]: constant offset
    - p[1:4]: linear coefficients for x, y, z
    - p[4:7]: RBF widths (log-space for stability)
    - p[7:10]: RBF weights for three fixed centers
    
    Inputs are in [0.5, 2]^3. RBF centers are placed at strategic points
    to cover the input domain.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Polynomial basis for global structure
    t0 = np.ones_like(x)
    t1 = x
    t2 = y
    t3 = z
    
    # RBF centers chosen to cover [0.5, 2]^3 domain
    # Center 1: low-low-low corner region
    cx1, cy1, cz1 = 0.8, 0.8, 0.8
    # Center 2: mid-mid-mid
    cx2, cy2, cz2 = 1.25, 1.25, 1.25
    # Center 3: high-high-high corner region
    cx3, cy3, cz3 = 1.8, 1.8, 1.8
    
    # Compute squared distances to each center
    dx1 = x - cx1; dy1 = y - cy1; dz1 = z - cz1
    dx2 = x - cx2; dy2 = y - cy2; dz2 = z - cz2
    dx3 = x - cx3; dy3 = y - cy3; dz3 = z - cz3
    
    r2_1 = dx1*dx1 + dy1*dy1 + dz1*dz1
    r2_2 = dx2*dx2 + dy2*dy2 + dz2*dz2
    r2_3 = dx3*dx3 + dy3*dy3 + dz3*dz3
    
    # RBF widths in log-space for numerical stability
    # Widths range from ~0.1 to ~1.0 (exp of -2.3 to 0)
    w1 = np.exp(np.clip(p[4], -3.0, 0.0))
    w2 = np.exp(np.clip(p[5], -3.0, 0.0))
    w3 = np.exp(np.clip(p[6], -3.0, 0.0))
    
    # Gaussian RBF values: exp(-r^2 / (2 * width^2))
    rbf1 = np.exp(-r2_1 / (2.0 * w1 * w1))
    rbf2 = np.exp(-r2_2 / (2.0 * w2 * w2))
    rbf3 = np.exp(-r2_3 / (2.0 * w3 * w3))
    
    # Combine all basis functions
    result = (p[0] * t0 + 
              p[1] * t1 + 
              p[2] * t2 + 
              p[3] * t3 + 
              p[7] * rbf1 + 
              p[8] * rbf2 + 
              p[9] * rbf3)
    
    return result
