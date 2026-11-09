"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Neural Network-like model with 2 hidden tanh units and linear passthrough.
    params: [w1_x, w1_y, w1_b, w2_x, w2_y, w2_b, out1, out2, lin_x, lin_y]
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    
    # Extract parameters
    w1_x, w1_y, w1_b, w2_x, w2_y, w2_b, out1, out2, lin_x, lin_y = params
    
    # Compute hidden units
    h1 = np.tanh(w1_x * x + w1_y * y + w1_b)
    h2 = np.tanh(w2_x * x + w2_y * y + w2_b)
    
    # Combine
    pred = out1 * h1 + out2 * h2 + lin_x * x + lin_y * y
    
    return pred
# EVOLVE-BLOCK-END
