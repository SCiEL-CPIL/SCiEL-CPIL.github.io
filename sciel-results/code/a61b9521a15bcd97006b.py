import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates a function using a linear slope and 9 Gaussian Radial Basis Functions.
    params[0]: linear slope
    params[1:10]: amplitudes for 9 RBFs
    """
    # Linear part (slope only, intercept handled by RBFs)
    y_linear = params[0] * x
    
    # RBF part
    # 9 centers to increase local resolution
    centers = np.linspace(-3.0, 3.0, 9)
    # Smaller width to capture sharper features
    sigma = 0.5
    
    # Compute distances
    x_expanded = x[:, np.newaxis]
    c_expanded = centers[np.newaxis, :]
    
    diff = x_expanded - c_expanded
    
    # Gaussian kernel
    rbf_values = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Weighted sum of RBFs
    y_rbf = rbf_values @ params[1:10]
    
    return y_linear + y_rbf
