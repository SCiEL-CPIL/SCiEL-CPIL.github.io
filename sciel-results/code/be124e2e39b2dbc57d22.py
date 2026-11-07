"""Symbolic regression with shifted sigmoid basis functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model using a sum of shifted sigmoid (logistic) basis functions.
    
    y = p0 + p1 * sigmoid(p2 * (x - p3)) + p4 * sigmoid(p5 * (x - p6))
        + p7 * sigmoid(p8 * (x - p9))
    
    params layout (length 10):
      p0: constant bias
      p1, p2, p3: amplitude, scale, center for first sigmoid
      p4, p5, p6: amplitude, scale, center for second sigmoid
      p7, p8, p9: amplitude, scale, center for third sigmoid
    
    The sigmoid function 1/(1+exp(-z)) provides smooth step-like transitions,
    which can model piecewise-constant or saturating behaviors that are hard
    to capture with polynomial or rational bases.
    """
    x = np.asarray(x, dtype=float)
    
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    # Compute sigmoid terms with numerical stability
    # sigmoid(z) = 1 / (1 + exp(-z))
    # For large positive z, sigmoid -> 1; for large negative z, sigmoid -> 0
    
    def sigmoid(z):
        # Clip z to avoid overflow in exp
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))
    
    s1 = sigmoid(p2 * (x - p3))
    s2 = sigmoid(p5 * (x - p6))
    s3 = sigmoid(p8 * (x - p9))
    
    result = p0 + p1 * s1 + p4 * s2 + p7 * s3
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
