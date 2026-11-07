import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize inputs to [0, 1]
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    t_norm = (temp - 1.0) / 39.0
    ph_norm = (pH - 2.0) / 10.0
    
    p = params
    
    # Composite 1D feature: Weighted sum of normalized inputs
    # Temp and pH are primary drivers; b and s are secondary nutrients
    # We project 4D space to 1D to allow 4 Gaussians within 10 params
    # z = 0.5*t + 0.3*ph + 0.1*b + 0.1*s
    z = 0.5 * t_norm + 0.3 * ph_norm + 0.1 * b_norm + 0.1 * s_norm
    
    # 4 Gaussian Basis Functions
    # Params:
    # p[0]: offset
    # p[1]: w1, p[2]: c1
    # p[3]: w2, p[4]: c2
    # p[5]: w3, p[6]: c3
    # p[7]: w4, p[8]: c4
    # p[9]: width_sq (shared isotropic width in 1D)
    
    width_sq = p[9]**2 + 1e-6
    
    # Calculate Gaussians
    # Using stable exponential calculation
    g1 = np.exp(-(z - p[2])**2 / (2.0 * width_sq))
    g2 = np.exp(-(z - p[4])**2 / (2.0 * width_sq))
    g3 = np.exp(-(z - p[6])**2 / (2.0 * width_sq))
    g4 = np.exp(-(z - p[8])**2 / (2.0 * width_sq))
    
    y = p[0] + p[1]*g1 + p[3]*g2 + p[5]*g3 + p[7]*g4
    
    if not np.all(np.isfinite(y)):
        return np.zeros_like(y)
        
    return y
