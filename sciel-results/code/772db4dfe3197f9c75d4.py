"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression candidate using a small neural network (2 hidden units, tanh activation).
    
    params is an array of length 10.
    p[0], p[1], p[2]: Weights and bias for hidden unit 1
    p[3], p[4], p[5]: Weights and bias for hidden unit 2
    p[6], p[7]: Output weights for hidden units
    p[8]: Weight for direct linear x term
    p[9]: Output bias
    
    Model:
    h1 = tanh(p[0]*x + p[1]*y + p[2])
    h2 = tanh(p[3]*x + p[4]*y + p[5])
    pred = p[6]*h1 + p[7]*h2 + p[8]*x + p[9]
    """
    # Ensure x and y are 1D arrays
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute hidden layers
    h1 = np.tanh(p0 * x + p1 * y + p2)
    h2 = np.tanh(p3 * x + p4 * y + p5)
    
    # Combine with output weights and linear term
    pred = p6 * h1 + p7 * h2 + p8 * x + p9
    
    return pred
# EVOLVE-BLOCK-END
