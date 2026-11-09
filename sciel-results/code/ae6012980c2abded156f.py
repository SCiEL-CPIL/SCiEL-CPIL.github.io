# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Equation using a linear combination of 10 orthogonal basis functions.
    The basis functions are scaled sine and cosine terms to approximate
    periodic or smooth functions on the domain [0.1, 6].
    
    params[0:5] are coefficients for cos(k * scaled_x)
    params[5:10] are coefficients for sin(k * scaled_x)
    
    This formulation is linear in parameters, ensuring the optimization
    landscape is convex (a paraboloid), so BFGS will find the global optimum.
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    
    # Scale x to a standard range for the trigonometric functions.
    # Domain is [0.1, 6]. Length is approx 5.9.
    # Let's map x to [0, 2*pi] or similar to use standard frequencies.
    # x_min = 0.1, x_max = 6.0
    # scaled_x = (x - 0.1) / (6.0 - 0.1) * 2 * pi
    # This maps [0.1, 6] to [0, 2*pi].
    
    x_min = 0.1
    x_max = 6.0
    scaled_x = (x - x_min) / (x_max - x_min) * 2.0 * np.pi
    
    # Generate basis functions
    # k = 1, 2, 3, 4, 5
    k = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    
    # Compute cos and sin terms
    # scaled_x shape: (N,)
    # k shape: (5,)
    # We need (N, 5)
    angles = np.outer(scaled_x, k)
    
    cos_terms = np.cos(angles)  # (N, 5)
    sin_terms = np.sin(angles)  # (N, 5)
    
    # Extract parameters
    w_cos = params[0:5]
    w_sin = params[5:10]
    
    # Compute prediction
    # y = sum(w_cos * cos_terms) + sum(w_sin * sin_terms)
    y_pred = np.sum(cos_terms * w_cos[np.newaxis, :], axis=1) + np.sum(sin_terms * w_sin[np.newaxis, :], axis=1)
    
    return y_pred
# EVOLVE-BLOCK-END
