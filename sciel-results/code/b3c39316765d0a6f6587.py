# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of 10 RBFs at fixed, evenly spaced centers.
    
    params layout (length 10):
      params[0]  : shared RBF width (log-scale for stability, exp-transformed)
      params[1:11]: coefficients for each of the 10 RBF centers
    
    Centers are fixed at evenly spaced points in [-3, 3]:
      centers = linspace(-3, 3, 10)
    
    The width is exp(params[0]) to keep it positive.
    """
    # Extract width parameter (positive via exp)
    log_width = params[0]
    # Clamp log_width to avoid numerical issues
    log_width = np.clip(log_width, -5.0, 5.0)
    width = np.exp(log_width)
    
    # Fixed centers evenly spaced in [-3, 3]
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Coefficients for each RBF
    coeffs = params[1:11]
    
    # Compute RBF values: sum_i coeffs[i] * exp(-((x - centers[i])^2) / (2 * width^2))
    # x shape: (n,), centers shape: (10,)
    # We want output shape: (n,)
    
    # diff shape: (n, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    # rbf shape: (n, 10)
    rbf = np.exp(-0.5 * (diff / width) ** 2)
    
    # Final prediction: (n, 10) @ (10,) -> (n,)
    prediction = rbf @ coeffs
    
    return prediction
# EVOLVE-BLOCK-END
