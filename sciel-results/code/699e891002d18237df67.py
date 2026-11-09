import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Radial Basis Function (RBF) model with 10 fixed centers and fixed width.
    
    Parameters (10 total):
      - params[0:10]: weights for each RBF center
    
    Centers are evenly spaced across [-3, 3].
    Width is fixed to 1.0.
    
    Each basis function is: w_i * exp(-((x - c_i)^2) / (2 * 1.0^2))
    """
    # 10 centers evenly spaced in [-3, 3]
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Fixed width
    sigma = 1.0
    
    # Compute RBF basis
    # x shape: (N,)
    # centers shape: (10,)
    # diff shape: (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # RBF values
    rbf_vals = np.exp(-(diff ** 2) / (2.0 * sigma ** 2))
    
    # Linear combination
    # params shape: (10,)
    result = rbf_vals @ params
    
    return result
