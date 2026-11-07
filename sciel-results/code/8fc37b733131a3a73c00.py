import numpy as np
import time

_start_time = time.time()
_DEADLINE = _start_time + 140.0

def equation(b, s, temp, pH, params):
    p = params
    
    # Check deadline
    if time.time() > _DEADLINE:
        return np.zeros_like(b)
    
    # Normalize inputs to [0, 1]
    b_norm = np.clip((b - 1.0) / (4.998 - 1.0), 0, 1)
    s_norm = np.clip((s - 1.0) / (4.999 - 1.0), 0, 1)
    temp_norm = np.clip((temp - 1.002) / (39.98 - 1.002), 0, 1)
    ph_norm = np.clip((pH - 2.0) / (12.0 - 2.0), 0, 1)
    
    # Strategy: Sparse sum of 4 Gaussian basis functions
    # Each Gaussian is centered in (temp_norm, ph_norm, s_norm) space
    # Parameters:
    # p[0]: base constant (global offset)
    # p[1]: shared isotropic width (sigma) for all Gaussians
    # p[2]: weight for Gaussian 1
    # p[3]: center temp for Gaussian 1
    # p[4]: center pH for Gaussian 1
    # p[5]: center s for Gaussian 1
    # p[6]: weight for Gaussian 2
    # p[7]: center temp for Gaussian 2
    # p[8]: center pH for Gaussian 2
    # p[9]: center s for Gaussian 2
    
    # Ensure width is positive and not too small to avoid numerical issues
    sigma = np.maximum(np.abs(p[1]), 0.01)
    
    # Gaussian 1
    d1 = (temp_norm - p[3])**2 + (ph_norm - p[4])**2 + (s_norm - p[5])**2
    g1 = np.exp(-d1 / (2.0 * sigma**2))
    
    # Gaussian 2
    d2 = (temp_norm - p[7])**2 + (ph_norm - p[8])**2 + (s_norm - p[9])**2
    g2 = np.exp(-d2 / (2.0 * sigma**2))
    
    # Linear terms for b_norm (substrate concentration) and a simple temp/pH interaction
    # to capture monotonic dependencies that Gaussians might miss
    result = (p[0] 
              + p[2] * g1 
              + p[6] * g2
              + b_norm * np.exp(-((temp_norm - 0.5)**2 + (ph_norm - 0.5)**2) / (2.0 * sigma**2)))
    
    return result
