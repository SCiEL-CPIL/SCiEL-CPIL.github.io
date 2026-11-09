import numpy as np
import time
from scipy.optimize import minimize

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression candidate.
    The evaluator fits params (length 10) via BFGS minimizing training MSE.
    We use a rich, diverse basis to approximate a wide range of functions.
    """
    # Ensure x is float array
    x = np.asarray(x, dtype=np.float64)
    
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute basis functions - diverse set to cover many possible target functions
    # Include polynomials, trigonometric, exponential, log, rational
    b0 = np.ones_like(x)
    b1 = x
    b2 = x**2
    b3 = x**3
    b4 = x**4
    
    # Trigonometric
    b5 = np.sin(x)
    b6 = np.cos(x)
    
    # Exponential and log
    b7 = np.exp(-x)  # decaying exponential (more stable than exp(x) for x in [0.1, 6])
    b8 = np.log(x)
    
    # Rational and other
    b9 = 1.0 / x
    
    # Linear combination
    result = (p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3 + p4 * b4 +
              p5 * b5 + p6 * b6 + p7 * b7 + p8 * b8 + p9 * b9)
    
    # Ensure finite values
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
