import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian RBF approximation with 10 fixed centers.
    Centers are evenly spaced in [-3, 3].
    params are the weights for each RBF.
    """
    # Fixed centers evenly spaced in [-3, 3]
    # Using linspace to cover the domain
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Fixed width for the Gaussians
    # A width of ~0.8 allows some overlap but maintains locality
    width = 0.8
    
    # Compute Gaussian values
    # x is shape (N,), centers is shape (10,)
    # diff is shape (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # exp(-0.5 * ((x - center) / width)^2)
    rbf_values = np.exp(-0.5 * (diff / width) ** 2)
    
    # Weighted sum
    # rbf_values (N, 10) @ params (10,) -> (N,)
    y = rbf_values @ params
    
    return y
