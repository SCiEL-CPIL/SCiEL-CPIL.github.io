# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    3-hidden-unit ReLU network with dedicated output weights and bias.
    
    Architecture:
    - Input: x (scalar)
    - Hidden layer: 3 ReLU units
    - Output: linear combination of hidden activations + bias
    
    Parameters (10 total):
    - params[0:3]: hidden layer weights (w_i)
    - params[3:6]: hidden layer biases (b_i)
    - params[6:9]: output weights (v_i)
    - params[9]: bias (c)
    
    Output: y = sum_i v_i * ReLU(w_i * x + b_i) + c
    This is a piecewise linear function with 3 kinks, fully parameterized.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    
    # Extract parameters
    w = params[0:3]   # hidden weights
    b = params[3:6]   # hidden biases
    v = params[6:9]   # output weights
    c = params[9]     # bias
    
    # Compute hidden activations: ReLU(w_i * x + b_i) for i=0..2
    # Shape: (n, 3)
    Z = np.zeros((n, 3), dtype=np.float64)
    for i in range(3):
        Z[:, i] = np.maximum(0.0, w[i] * x + b[i])
    
    # Output: weighted sum of hidden activations + bias
    pred = Z @ v + c
    
    return pred
# EVOLVE-BLOCK-END
