import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Params layout:
    # 0-2: Center 1 (cx1, cy1, cz1)
    # 3-5: Center 2 (cx2, cy2, cz2)
    # 6: Weight 1 (w1)
    # 7: Weight 2 (w2)
    # 8: Width parameter (log_sigma) -> sigma = exp(log_sigma)
    # 9: Bias (b)
    
    cx1, cy1, cz1 = params[0], params[1], params[2]
    cx2, cy2, cz2 = params[3], params[4], params[5]
    w1, w2 = params[6], params[7]
    log_sigma = params[8]
    b = params[9]
    
    sigma = np.exp(log_sigma)
    
    # Calculate squared distances
    d1_sq = (x - cx1)**2 + (y - cy1)**2 + (z - cz1)**2
    d2_sq = (x - cx2)**2 + (y - cy2)**2 + (z - cz2)**2
    
    # RBF terms
    rbf1 = np.exp(-d1_sq / (2 * sigma**2))
    rbf2 = np.exp(-d2_sq / (2 * sigma**2))
    
    return w1 * rbf1 + w2 * rbf2 + b
