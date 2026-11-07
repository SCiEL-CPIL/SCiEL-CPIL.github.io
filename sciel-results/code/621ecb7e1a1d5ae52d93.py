import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Fixed centers evenly spaced in [-2, 2]
    centers = np.linspace(-2.0, 2.0, 10)
    
    # Fixed width
    # Spacing is 4/9 ~ 0.44. 
    # A width of ~0.5-0.8 usually works well for smooth functions.
    # Let's use 0.6.
    sigma = 0.6
    
    # Calculate Gaussian basis functions
    # x shape: (N,)
    # centers shape: (10,)
    # diff shape: (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Exponent term
    # exp(-0.5 * (diff/sigma)^2)
    basis = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Linear combination with params
    # basis shape: (N, 10)
    # params shape: (10,)
    predictions = basis @ params
    
    return predictions
