import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates a function using a quadratic trend and 7 Gaussian Radial Basis Functions.
    params[0]: quadratic coefficient
    params[1]: linear coefficient
    params[2]: intercept
    params[3:10]: amplitudes for 7 RBFs
    """
    # Quadratic trend part
    y_poly = params[0] * x**2 + params[1] * x + params[2]
    
    # RBF part
    # Fixed centers
    centers = np.linspace(-3.0, 3.0, 7)
    # Fixed width
    sigma = 0.8
    
    # Compute distances
    x_expanded = x[:, np.newaxis]
    c_expanded = centers[np.newaxis, :]
    
    diff = x_expanded - c_expanded
    
    # Gaussian kernel
    rbf_values = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Weighted sum of RBFs
    y_rbf = rbf_values @ params[3:10]
    
    return y_poly + y_rbf
