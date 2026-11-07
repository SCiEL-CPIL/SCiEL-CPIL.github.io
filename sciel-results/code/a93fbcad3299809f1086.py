import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Radial Basis Function (RBF) network.
    
    The model uses 10 parameters:
    - 5 centers c_i (params[0:5])
    - 5 widths sigma_i (params[5:10])
    
    The output is the sum of 5 Gaussian kernels:
    y(x) = sum_i exp(-(x - c_i)^2 / (2 * sigma_i^2))
    
    This shifts from global polynomial approximation to local feature interpolation,
    allowing the model to independently adjust the location and scale of localized
    features in the hidden function.
    """
    # Extract centers and widths
    centers = params[0:5]  # c_1, ..., c_5
    widths = params[5:10]  # sigma_1, ..., sigma_5
    
    # Clip widths to avoid division by zero or extremely small values
    # Using a soft clip to maintain gradient flow
    sigma = np.maximum(np.abs(widths), 0.01)
    
    # Compute RBF output: sum of Gaussian kernels
    # y(x) = sum_i exp(-(x - c_i)^2 / (2 * sigma_i^2))
    y = np.zeros_like(x, dtype=float)
    
    for i in range(5):
        diff = x - centers[i]
        y += np.exp(-(diff * diff) / (2.0 * sigma[i] * sigma[i]))
    
    return y
