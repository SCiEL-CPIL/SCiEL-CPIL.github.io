import numpy as np
from numpy.polynomial import chebyshev as C

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a Chebyshev series of degree 9.
    The input x is assumed to be in [-2, 2]. We map it to [-1, 1] for the
    Chebyshev polynomials.
    
    Model: y(x) = sum_{i=0}^{9} params[i] * T_i(x/2)
    
    This is a linear model in the parameters, which ensures that BFGS
    (or any gradient-based optimizer) will converge to the global minimum
    of the least squares problem efficiently and robustly.
    """
    # Map x from [-2, 2] to [-1, 1]
    t = x / 2.0
    
    # Evaluate Chebyshev series
    # params is an array of length 10, corresponding to coefficients for T_0 to T_9
    return C.chebval(t, params)
