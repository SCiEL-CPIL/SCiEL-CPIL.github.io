import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # RBF + polynomial model
    # 5 Gaussian RBF centers + 5 polynomial terms
    # Centers: (-1,-1), (1,-1), (-1,1), (1,1), (0,0)
    cx = np.array([-1.0, 1.0, -1.0, 1.0, 0.0])
    cy = np.array([-1.0, -1.0, 1.0, 1.0, 0.0])
    
    # Tighter Gaussian width for more local flexibility
    sigma = 0.6
    
    # Compute Gaussian kernels for each center
    # For efficiency, compute squared distances
    b_gauss = np.zeros_like(x)
    for i in range(5):
        dx = x - cx[i]
        dy = y - cy[i]
        dist_sq = dx**2 + dy**2
        b_gauss += params[i] * np.exp(-dist_sq / (2.0 * sigma**2))
    
    # Polynomial terms: constant, x, y, x^2, y^2
    b_poly = params[5] + params[6] * x + params[7] * y + params[8] * x**2 + params[9] * y**2
    
    pred = b_gauss + b_poly
    
    return pred
