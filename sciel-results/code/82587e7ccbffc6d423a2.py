import numpy as np
import time

# Global start time for deadline guard
_start_time = time.time()
_DEADLINE = 140.0  # Safety margin from 150s

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        # Return a simple fallback to avoid crash, though this shouldn't happen in forward pass
        return np.zeros_like(x)

    # Params layout:
    # 0-2: Center 1 (cx1, cy1, cz1)
    # 3-5: Center 2 (cx2, cy2, cz2)
    # 6: Weight 1 (w1)
    # 7: Weight 2 (w2)
    # 8: Log-Width 1 (log_sigma1)
    # 9: Log-Width 2 (log_sigma2)
    
    cx1, cy1, cz1 = params[0], params[1], params[2]
    cx2, cy2, cz2 = params[3], params[4], params[5]
    w1, w2 = params[6], params[7]
    log_sigma1 = params[8]
    log_sigma2 = params[9]
    
    sigma1 = np.exp(log_sigma1)
    sigma2 = np.exp(log_sigma2)
    
    # Calculate squared distances
    d1_sq = (x - cx1)**2 + (y - cy1)**2 + (z - cz1)**2
    d2_sq = (x - cx2)**2 + (y - cy2)**2 + (z - cz2)**2
    
    # RBF terms
    rbf1 = np.exp(-d1_sq / (2 * sigma1**2))
    rbf2 = np.exp(-d2_sq / (2 * sigma2**2))
    
    return w1 * rbf1 + w2 * rbf2
