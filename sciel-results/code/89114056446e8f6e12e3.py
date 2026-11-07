import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Define 10 fixed Gaussian centers in [-2, 2]
    centers = np.linspace(-2.0, 2.0, 10)
    
    # Width parameter for the Gaussians
    sigma = 0.5
    
    # Compute basis functions
    # x is shape (N,), centers is shape (10,)
    # We want P to be shape (N, 10) so that y = P @ params
    
    x_expanded = x[:, np.newaxis]
    centers_expanded = centers[np.newaxis, :]
    
    diff = x_expanded - centers_expanded
    sq_dist = diff * diff
    
    y = np.exp(-sq_dist / (2.0 * sigma * sigma))
    
    prediction = y @ params
    
    return prediction
