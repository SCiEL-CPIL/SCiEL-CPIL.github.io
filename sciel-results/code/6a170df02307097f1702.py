# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Gaussian RBF model with 5 fixed centers and learned weights/widths
    # Centers evenly spaced across [-3, 3]
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    
    # params[0:5] = weights for each RBF
    # params[5:10] = width parameters (positive via softplus-like transform)
    
    weights = params[0:5]
    # Transform width params to positive values: width = 0.1 + exp(params[5:10])
    # This ensures widths are always positive and well-conditioned
    widths = 0.1 + np.exp(np.clip(params[5:10], -10, 10))
    
    # Compute RBF outputs: sum of weights * exp(-((x - center)^2) / (2 * width^2))
    result = np.zeros_like(x, dtype=np.float64)
    
    for i in range(5):
        diff = x - centers[i]
        # Use width as standard deviation, so variance = width^2
        exponent = -(diff * diff) / (2.0 * widths[i] * widths[i])
        # Clip exponent to avoid underflow issues
        exponent = np.clip(exponent, -500, 500)
        result += weights[i] * np.exp(exponent)
    
    return result
# EVOLVE-BLOCK-END
