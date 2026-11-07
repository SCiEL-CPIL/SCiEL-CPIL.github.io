import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map x from [0.1, 6.0] to z in [-1, 1]
    x_min = 0.1
    x_max = 6.0
    z = (2.0 * x - (x_max + x_min)) / (x_max - x_min)
    
    # Clip z to [-1, 1] to avoid numerical issues with arccos
    z = np.clip(z, -1.0, 1.0)
    
    # Compute angles
    theta = np.arccos(z)
    
    # Compute Chebyshev basis T_k(z) = cos(k * theta)
    # k = 0 to 9
    k = np.arange(10)[:, np.newaxis]
    theta_row = theta[np.newaxis, :]
    
    basis = np.cos(k * theta_row)
    
    # Linear combination
    return basis @ params
