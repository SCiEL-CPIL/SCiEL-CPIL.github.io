import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    N = len(x)
    
    # Basis 0: Constant
    # Basis 1: Linear
    # Basis 2: Quadratic
    # Basis 3-9: 7 Gaussians
    
    P = np.zeros((10, N))
    
    P[0, :] = 1.0
    P[1, :] = x
    P[2, :] = x * x
    
    # Gaussian centers spread across the domain
    centers = np.linspace(-1.8, 1.8, 7)
    sigma = 0.8 # Slightly narrower to allow for some local variation, but still overlapping
    
    # Calculate Gaussians
    diff = x[:, None] - centers[None, :]
    gaussians = np.exp(-(diff ** 2) / (2 * sigma ** 2))
    
    P[3:, :] = gaussians
    
    # Linear combination
    y = np.dot(params, P)
    
    return y
