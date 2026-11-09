import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the prediction using a fixed Gaussian RBF basis + polynomial trend.
    
    Args:
        x: Input array of shape (N,)
        params: Coefficients array of shape (10,)
                 params[0]: constant term
                 params[1]: linear term
                 params[2]: quadratic term
                 params[3:10]: RBF amplitudes
    
    Returns:
        Prediction array of shape (N,)
    """
    x = np.asarray(x).ravel()
    N = len(x)
    
    # 1. Polynomial trend (degrees 0, 1, 2)
    # This captures the global shape
    poly_part = params[0] * 1.0 + params[1] * x + params[2] * x**2
    
    # 2. Gaussian RBFs
    # Fixed centers spread across [-3, 3]
    # We have 7 RBFs (indices 3 to 9 in params)
    centers = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 0.0])
    # Fixed widths (standard deviations)
    # Wider widths for smoother functions, narrower for sharper features.
    # A mix of widths helps robustness.
    widths = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1.2])
    
    rbf_part = np.zeros(N)
    for i in range(7):
        # Compute Gaussian: exp(-0.5 * ((x - center) / width)^2)
        # Using 0.5 * ((x-c)/w)^2 is equivalent to exp(-(x-c)^2 / (2*w^2))
        diff = x - centers[i]
        rbf_vals = np.exp(-0.5 * (diff / widths[i])**2)
        rbf_part += params[3 + i] * rbf_vals
        
    return poly_part + rbf_part
