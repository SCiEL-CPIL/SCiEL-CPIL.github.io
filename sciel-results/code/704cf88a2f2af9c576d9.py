"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Universal approximator using a small neural network.
    
    We have 10 parameters total. We use a fixed architecture:
    - 2 hidden units (to stay within 10 params: 2 weights in, 2 biases, 2 weights out, 1 bias = 7... 
      Wait, let's count carefully.)
      
    Let's design a network that fits exactly 10 parameters.
    Architecture:
    Input: x
    Hidden layer: 2 units
    Output layer: 1 unit
    
    Parameters:
    W1 (2x1): 2 params
    b1 (2):   2 params
    W2 (1x2): 2 params
    b2 (1):   1 param
    Total so far: 7 params.
    
    We have 3 extra parameters. We can expand the hidden layer to 4 units?
    W1 (4x1): 4 params
    b1 (4):   4 params
    W2 (1x4): 4 params
    b2 (1):   1 param
    Total: 13 params. Too many.
    
    Let's try 3 hidden units:
    W1 (3x1): 3 params
    b1 (3):   3 params
    W2 (1x3): 3 params
    b2 (1):   1 param
    Total: 10 params. Perfect.
    
    So:
    h1 = ReLU(w1[0]*x + b1[0])
    h2 = ReLU(w1[1]*x + b1[0])  # Wait, indices
    h3 = ReLU(w1[2]*x + b1[2])
    
    Let's map params:
    params[0:3] -> W1 (weights from input to hidden)
    params[3:6] -> b1 (biases for hidden)
    params[6:9] -> W2 (weights from hidden to output)
    params[9]   -> b2 (bias for output)
    
    Computation:
    z1 = params[0]*x + params[3]
    z2 = params[1]*x + params[4]
    z3 = params[2]*x + params[5]
    
    a1 = ReLU(z1)
    a2 = ReLU(z2)
    a3 = ReLU(z3)
    
    y = params[6]*a1 + params[7]*a2 + params[8]*a3 + params[9]
    """
    # Extract parameters
    w1_0, w1_1, w1_2 = params[0], params[1], params[2]
    b1_0, b1_1, b1_2 = params[3], params[4], params[5]
    w2_0, w2_1, w2_2 = params[6], params[7], params[8]
    b2 = params[9]
    
    # Hidden layer computations
    z1 = w1_0 * x + b1_0
    z2 = w1_1 * x + b1_1
    z3 = w1_2 * x + b1_2
    
    # ReLU activation
    a1 = np.maximum(0, z1)
    a2 = np.maximum(0, z2)
    a3 = np.maximum(0, z3)
    
    # Output layer
    y = w2_0 * a1 + w2_1 * a2 + w2_2 * a3 + b2
    
    return y
# EVOLVE-BLOCK-END
