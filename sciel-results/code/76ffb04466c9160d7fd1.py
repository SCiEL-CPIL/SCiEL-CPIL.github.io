import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a rich basis of polynomial, trigonometric,
    and exponential/rational terms to cover a wide function class.
    
    We use 10 parameters for:
    - 4 polynomial terms: 1, x, x^2, x^3
    - 4 trigonometric terms: sin(x), cos(x), sin(2x), cos(2x)
    - 2 exponential/rational terms: exp(-x^2), 1/(1+x^2)
    
    This basis captures polynomials, periodic functions, Gaussian-like bumps,
    and Lorentzian shapes, providing good coverage of common hidden functions.
    """
    x = np.asarray(x, dtype=np.float64)
    n_points = x.size
    
    # Initialize the basis matrix
    # Shape: (n_points, 10)
    V = np.zeros((n_points, 10), dtype=np.float64)
    
    # Polynomial terms (indices 0-3)
    V[:, 0] = 1.0
    V[:, 1] = x
    V[:, 2] = x * x
    V[:, 3] = x * x * x
    
    # Trigonometric terms (indices 4-7)
    V[:, 4] = np.sin(x)
    V[:, 5] = np.cos(x)
    V[:, 6] = np.sin(2.0 * x)
    V[:, 7] = np.cos(2.0 * x)
    
    # Exponential/rational terms (indices 8-9)
    V[:, 8] = np.exp(-x * x)
    V[:, 9] = 1.0 / (1.0 + x * x)
    
    # Compute the linear combination
    # params shape: (10,)
    # V shape: (n_points, 10)
    # Result shape: (n_points,)
    return V @ params
