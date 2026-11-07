# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Neural network with tanh activation: 2 inputs -> 3 hidden -> 1 output.
    
    Parameters (10 total):
      p[0..5]: weights for 2->3 hidden layer (w11, w12, w13, w21, w22, w23)
      p[6..8]: weights for 3->1 output layer (v1, v2, v3)
      p[9]:    output bias
    
    Hidden layer: h_i = tanh(sum_j w_ji * x_j) for i=1,2,3
    Output:       y_pred = sum_i v_i * h_i + bias
    
    This is a universal approximator, far more expressive than 10 linear basis functions.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Hidden layer: 3 neurons, tanh activation
    # Input vector: [x, y]
    h1 = np.tanh(p[0] * x + p[3] * y)
    h2 = np.tanh(p[1] * x + p[4] * y)
    h3 = np.tanh(p[2] * x + p[5] * y)
    
    # Output layer
    return p[6] * h1 + p[7] * h2 + p[8] * h3 + p[9]
# EVOLVE-BLOCK-END
