import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Radial Basis Function (RBF) model with Gaussian kernels
    # Centers at four corners of [-1,1]^2 and the origin
    # 10 parameters: 5 for Gaussian weights, 5 for linear terms
    
    # Centers: (-1,-1), (1,-1), (-1,1), (1,1), (0,0)
    cx = np.array([-1.0, 1.0, -1.0, 1.0, 0.0])
    cy = np.array([-1.0, -1.0, 1.0, 1.0, 0.0])
    
    # Gaussian width parameter (fixed for stability)
    sigma = 0.8
    
    # Compute Gaussian kernels for each center
    # For efficiency, compute squared distances
    b_gauss = np.zeros_like(x)
    for i in range(5):
        dx = x - cx[i]
        dy = y - cy[i]
        dist_sq = dx**2 + dy**2
        b_gauss += params[i] * np.exp(-dist_sq / (2.0 * sigma**2))
    
    # Linear terms for better global approximation
    b_lin = params[5] * x + params[6] * y + params[7] * x * y + params[8] * x**2 + params[9] * y**2
    
    pred = b_gauss + b_lin
    
    return pred
