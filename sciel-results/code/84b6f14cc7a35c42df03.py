"""Rewritten symbolic regression candidate with a richer basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a fixed set of 10 basis functions.
    params: length-10 array of coefficients (fitted by BFGS).
    Returns an array with one prediction per point.
    """
    # Ensure inputs are arrays and flatten to 1D for consistent shape
    x = np.asarray(x, dtype=float).flatten()
    y = np.asarray(y, dtype=float).flatten()
    
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Compute basis functions
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = x**2
    b4 = y**2
    b5 = x * y
    b6 = np.sin(x)
    b7 = np.cos(y)
    b8 = np.tanh(x)
    b9 = np.tanh(y)
    
    # Linear combination
    pred = (p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3 + p4 * b4 + 
            p5 * b5 + p6 * b6 + p7 * b7 + p8 * b8 + p9 * b9)
            
    return pred
# EVOLVE-BLOCK-END
