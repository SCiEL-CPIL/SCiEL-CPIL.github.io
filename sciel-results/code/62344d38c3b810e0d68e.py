import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # RBF model with 5 centers: 4 corners and origin
    # 10 params: [w0, log_gamma0, w1, log_gamma1, w2, log_gamma2, w3, log_gamma3, w4, log_gamma4]
    
    # Centers
    cx = np.array([-1.0, 1.0, -1.0, 1.0, 0.0])
    cy = np.array([-1.0, -1.0, 1.0, 1.0, 0.0])
    
    p = np.asarray(params).flatten()
    
    # Handle potential edge cases in param shape
    if p.size != 10:
        return np.zeros_like(x)
        
    weights = p[0:5:2]
    log_gammas = p[1:6:2]
    
    # Compute gammas
    gammas = np.exp(log_gammas)
    
    # Vectorized distance calculation
    # x, y are 1D arrays of length N
    dx = x[:, None] - cx[None, :]
    dy = y[:, None] - cy[None, :]
    dist_sq = dx**2 + dy**2
    
    # Kernel matrix: (N, 5)
    # Multiply by gammas: (N, 5) * (5,) -> (N, 5)
    kernel = np.exp(-dist_sq * gammas[None, :])
    
    # Prediction: (N, 5) dot (5,) -> (N,)
    pred = kernel @ weights
    
    return pred
