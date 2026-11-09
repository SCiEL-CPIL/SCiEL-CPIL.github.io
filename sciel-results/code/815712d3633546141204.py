import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a rich basis of polynomial, trigonometric,
    and exponential/rational terms to cover a wide function class.
    
    We use 10 parameters for:
    - 5 polynomial terms: 1, x, x^2, x^3, x^4
    - 3 trigonometric terms: sin(x), cos(x), sin(2x)
    - 1 exponential term: exp(-x^2)
    - 1 rational term: 1/(1+x^2)
    
    This basis captures polynomials up to degree 4, periodic functions,
    Gaussian-like bumps, and Lorentzian shapes, providing good coverage
    of common hidden functions while remaining well-conditioned for BFGS.
    """
    x = np.asarray(x, dtype=np.float64)
    n_points = x.size
    
    # Initialize the basis matrix
    # Shape: (n_points, 10)
    V = np.zeros((n_points, 10), dtype=np.float64)
    
    # Polynomial terms (indices 0-4): 1, x, x^2, x^3, x^4
    V[:, 0] = 1.0
    V[:, 1] = x
    V[:, 2] = x * x
    V[:, 3] = x * x * x
    V[:, 4] = x * x * x * x
    
    # Trigonometric terms (indices 5-7): sin(x), cos(x), sin(2x)
    V[:, 5] = np.sin(x)
    V[:, 6] = np.cos(x)
    V[:, 7] = np.sin(2.0 * x)
    
    # Exponential term (index 8): exp(-x^2)
    V[:, 8] = np.exp(-x * x)
    
    # Rational term (index 9): 1/(1+x^2)
    V[:, 9] = 1.0 / (1.0 + x * x)
    
    # Compute the linear combination
    # params shape: (10,)
    # V shape: (n_points, 10)
    # Result shape: (n_points,)
    return V @ params
