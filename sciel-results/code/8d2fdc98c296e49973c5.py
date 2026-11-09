"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the prediction using a linear combination of 10 basis functions.
    The model is linear in parameters, allowing the evaluator's BFGS optimizer
    to converge to the global least-squares optimum efficiently.
    
    Basis functions:
    1, x, x^2, x^3, x^4, x^5, sin(x), cos(x), sin(2x), cos(2x)
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Compute basis functions
    p0 = np.ones_like(x)
    p1 = x
    p2 = x**2
    p3 = x**3
    p4 = x**4
    p5 = x**5
    p6 = np.sin(x)
    p7 = np.cos(x)
    p8 = np.sin(2.0 * x)
    p9 = np.cos(2.0 * x)
    
    # Linear combination
    # Using dot product or explicit sum for clarity and speed
    # Explicit sum is often faster for small N basis functions than building a matrix
    return (params[0] * p0 + 
            params[1] * p1 + 
            params[2] * p2 + 
            params[3] * p3 + 
            params[4] * p4 + 
            params[5] * p5 + 
            params[6] * p6 + 
            params[7] * p7 + 
            params[8] * p8 + 
            params[9] * p9)
# EVOLVE-BLOCK-END
