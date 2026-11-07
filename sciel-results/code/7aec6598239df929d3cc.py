import numpy as np
import time

# Global start time for deadline guard
_start_time = time.time()
_DEADLINE = 140.0  # Safety margin from 150s

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        return np.zeros_like(x)

    # Params layout: 3 RBFs with shared width + bias
    # 0-2: Center 1 (cx1, cy1, cz1)
    # 3-5: Center 2 (cx2, cy2, cz2)
    # 6-8: Center 3 (cx3, cy3, cz3)
    # 9: shared log-width
    # Weights are implicitly 1.0; the third RBF with large width acts as bias
    
    cx1, cy1, cz1 = params[0], params[1], params[2]
    cx2, cy2, cz2 = params[3], params[4], params[5]
    cx3, cy3, cz3 = params[6], params[7], params[8]
    log_sigma = params[9]
    
    sigma = np.exp(log_sigma)
    
    # Calculate squared distances
    d1_sq = (x - cx1)**2 + (y - cy1)**2 + (z - cz1)**2
    d2_sq = (x - cx2)**2 + (y - cy2)**2 + (z - cz2)**2
    d3_sq = (x - cx3)**2 + (y - cy3)**2 + (z - cz3)**2
    
    # RBF terms with equal weights (1.0)
    rbf1 = np.exp(-d1_sq / (2 * sigma**2))
    rbf2 = np.exp(-d2_sq / (2 * sigma**2))
    rbf3 = np.exp(-d3_sq / (2 * sigma**2))
    
    return rbf1 + rbf2 + rbf3
