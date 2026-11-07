# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 2 inputs x, y in [-1, 1]^2.
    
    Strategy: Use a small 2-layer neural network with tanh activation.
    This is a structural change from a fixed linear basis to a learned
    feature transformation. The 10 parameters are:
      - Layer 1 (2 hidden units): 2 weights each (w1_x, w1_y) + 2 biases = 6 params
      - Layer 2 (1 output): 2 weights (w2_1, w2_2) + 1 bias = 3 params
      - 1 global scale factor
    Total: 10 params
    
    This lets BFGS shape the internal representations, not just scale fixed features.
    """
    p = params
    
    # Unpack parameters
    # Layer 1: 2 hidden units
    w1_1_x, w1_1_y, b1_1 = p[0], p[1], p[2]   # hidden unit 1
    w1_2_x, w1_2_y, b1_2 = p[3], p[4], p[5]   # hidden unit 2
    
    # Layer 2: output
    w2_1, w2_2, b2 = p[6], p[7], p[8]
    
    # Global scale
    scale = p[9]
    
    # Compute hidden layer
    h1 = np.tanh(w1_1_x * x + w1_1_y * y + b1_1)
    h2 = np.tanh(w1_2_x * x + w1_2_y * y + b1_2)
    
    # Output layer
    out = w2_1 * h1 + w2_2 * h2 + b2
    
    # Apply global scale
    return scale * out
# EVOLVE-BLOCK-END
