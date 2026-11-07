import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a 2-kernel Gaussian RBF network.
    Parameter allocation (10 total):
      p[0], p[1]: kernel weights
      p[2], p[3], p[4]: center of kernel 1 (cx1, cy1, cz1)
      p[5], p[6], p[7]: center of kernel 2 (cx2, cy2, cz2)
      p[8], p[9]: widths (sigmas) of kernels 1 and 2
    
    Each kernel is a 3D Gaussian: w_i * exp(-||[x,y,z] - c_i||^2 / (2*sigma_i^2))
    This provides flexible local approximation for non-polynomial features.
    """
    p = np.asarray(params, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    
    # Extract parameters
    w1 = p[0]
    w2 = p[1]
    cx1, cy1, cz1 = p[2], p[3], p[4]
    cx2, cy2, cz2 = p[5], p[6], p[7]
    s1 = p[8]
    s2 = p[9]
    
    # Smooth positive width parameterization
    # Clamp log-sigma to avoid extreme values: sigma in [exp(-3), exp(3)] ~ [0.05, 20]
    log_s1 = np.clip(p[8], -3.0, 3.0)
    log_s2 = np.clip(p[9], -3.0, 3.0)
    s1 = np.exp(log_s1)
    s2 = np.exp(log_s2)
    
    # Compute squared distances to each center
    d1 = (x - cx1)**2 + (y - cy1)**2 + (z - cz1)**2
    d2 = (x - cx2)**2 + (y - cy2)**2 + (z - cz2)**2
    
    # Gaussian kernels
    g1 = np.exp(-d1 / (2.0 * s1**2))
    g2 = np.exp(-d2 / (2.0 * s2**2))
    
    # Linear combination of kernels
    result = w1 * g1 + w2 * g2
    
    return result
