"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    A 2-hidden-unit MLP with direct linear connections.
    params:
    p0, p1, p2: weights and bias for hidden unit 1
    p3, p4, p5: weights and bias for hidden unit 2
    p6, p7: weights for hidden units in output
    p8, p9: weights for direct linear inputs x and y
    """
    # Extract parameters
    w11, w12, b1 = params[0], params[1], params[2]
    w21, w22, b2 = params[3], params[4], params[5]
    w_out1, w_out2 = params[6], params[7]
    w_x, w_y = params[8], params[9]
    
    # Compute hidden layers
    h1 = np.tanh(w11 * x + w12 * y + b1)
    h2 = np.tanh(w21 * x + w22 * y + b2)
    
    # Compute output
    pred = w_out1 * h1 + w_out2 * h2 + w_x * x + w_y * y
    
    return pred
# EVOLVE-BLOCK-END
