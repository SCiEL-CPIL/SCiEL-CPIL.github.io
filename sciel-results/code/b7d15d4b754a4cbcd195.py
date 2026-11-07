# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Small Neural Network with 3 hidden neurons (tanh activation).
    10 parameters total:
    p[0:3]: Input to Hidden weights
    p[3:6]: Hidden biases
    p[6:9]: Hidden to Output weights
    p[9]: Output bias
    
    Symmetry breaking shifts are applied to the weights and biases
    to prevent the initial all-ones state from causing identical
    hidden neuron activations, which hinders optimization.
    """
    p = params
    x = np.asarray(x, dtype=np.float64)
    orig_shape = x.shape
    x_flat = x.ravel()
    
    # Extract parameters
    w1 = p[0]
    w2 = p[1] + 0.1
    w3 = p[2] - 0.1
    
    b1 = p[3]
    b2 = p[4] + 0.1
    b3 = p[5] - 0.1
    
    v1 = p[6]
    v2 = p[7] + 0.1
    v3 = p[8] - 0.1
    
    b_out = p[9]
    
    # Hidden layer
    h1 = np.tanh(w1 * x_flat + b1)
    h2 = np.tanh(w2 * x_flat + b2)
    h3 = np.tanh(w3 * x_flat + b3)
    
    # Output layer
    result = v1 * h1 + v2 * h2 + v3 * h3 + b_out
    
    return result.reshape(orig_shape)
# EVOLVE-BLOCK-END
