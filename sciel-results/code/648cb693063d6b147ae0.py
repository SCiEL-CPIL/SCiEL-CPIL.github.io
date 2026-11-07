"""Symbolic regression with 1 input; Single-hidden-layer Neural Network with 10 parameters."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Single hidden layer neural network with 10 parameters.
    
    Architecture:
    - 8 hidden units with ReLU activation
    - 1 linear term (x)
    - 1 bias term
    
    Parameters (10 total):
    - p[0:8]: hidden layer weights (8)
    - p[8]: linear coefficient for x
    - p[9]: bias term
    
    Forward pass:
    h_i = ReLU(w_i * x) for i in 0..7
    y = sum(v_i * h_i) + v_linear * x + b
    
    Wait, I need 8 weights for hidden units AND 8 output weights for hidden units.
    That's 16 parameters just for the hidden layer. I only have 10.
    
    Let me redesign:
    - 6 hidden units with ReLU
    - 6 hidden weights (input to hidden)
    - 6 output weights (hidden to output)
    - 1 linear term weight
    - 1 bias
    Total: 6 + 6 + 1 + 1 = 14. Still too many.
    
    Alternative:
    - 4 hidden units with tanh
    - 4 input weights
    - 4 output weights  
    - 1 linear term
    - 1 bias
    Total: 4 + 4 + 1 + 1 = 10. Perfect!
    
    y = v1*tanh(w1*x) + v2*tanh(w2*x) + v3*tanh(w3*x) + v4*tanh(w4*x) + m*x + b
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Extract parameters
    w1, w2, w3, w4 = params[0], params[1], params[2], params[3]
    v1, v2, v3, v4 = params[4], params[5], params[6], params[7]
    m = params[8]
    b = params[9]
    
    # Compute hidden activations (tanh)
    h1 = np.tanh(w1 * x)
    h2 = np.tanh(w2 * x)
    h3 = np.tanh(w3 * x)
    h4 = np.tanh(w4 * x)
    
    # Output
    y = v1 * h1 + v2 * h2 + v3 * h3 + v4 * h4 + m * x + b
    
    return y
# EVOLVE-BLOCK-END
