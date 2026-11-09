import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF model with 9 fixed centers and 1 learnable width.
    x is mapped to [0, 1].
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 0:
        x = x.reshape(1)
        
    # Normalize x to [0, 1]
    t = (x - 0.1) / 5.9
    
    # Fixed centers in normalized space
    centers = np.linspace(0.0, 1.0, 9)
    
    # Learnable width (inverse variance scale)
    # Map p9 to positive sigma
    sigma = np.exp(params[9])
    
    # Weights for the 9 RBFs
    weights = params[:9]
    
    # Compute squared distances
    # t shape (N,), centers shape (9,)
    diff = t[:, np.newaxis] - centers[np.newaxis, :]
    
    # Gaussian kernel
    # Add a small epsilon to sigma to avoid division by zero if exp underflows?
    # exp(params[9]) is always > 0. If params[9] is very negative, sigma is very small.
    # diff^2 / (2 * sigma^2) will be huge, exp will be 0. This is fine.
    kernel = np.exp(-diff**2 / (2.0 * sigma**2))
    
    # Linear combination
    y = kernel @ weights
    
    return y
