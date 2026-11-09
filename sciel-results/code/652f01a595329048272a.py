"""
Strategy: Small ReLU Neural Network (1-5-1) with 10 Parameters.

The target function is unknown. We use a 2-layer ReLU network with 5 hidden units.
The 10 parameters are the weights of the first layer (5 weights + 5 biases).
The second layer has fixed weights and bias to allow for rich output combinations.
This structure provides a flexible non-linear basis that BFGS can optimize effectively.
"""

# EVOLVE-BLOCK-START
import numpy as np
import time

START_TIME = time.time()
DEADLINE = START_TIME + 145.0

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    1-5-1 ReLU Network.
    params[0:5] are weights for input to hidden layer.
    params[5:10] are biases for hidden layer.
    Hidden layer: h = ReLU(W1 * x + b1)
    Output layer: y = W2 * h + b2
    W2 and b2 are fixed to allow diverse combinations of hidden units.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Reshape if necessary, though x is usually 1D
    if x.ndim != 1:
        x = x.flatten()
        
    # Extract parameters
    W1 = params[0:5]  # Weights
    b1 = params[5:10] # Biases
    
    # Hidden layer computation
    # h_i = ReLU(W1[i] * x + b1[i])
    # To avoid creating large intermediate arrays unnecessarily, we compute directly
    # But for clarity and vectorization:
    
    z = W1 * x + b1  # Broadcasting: (5,) * (N,) + (5,) -> (5, N)
    h = np.maximum(z, 0.0) # ReLU
    
    # Fixed second layer weights and bias
    # We want the output to be a linear combination of the 5 hidden units.
    # If we set W2 to all 1s and b2 to 0, we get sum(h).
    # This is a very simple model. Can we do better with fixed weights?
    # Maybe alternating signs or different magnitudes?
    # Let's try to make the output layer slightly more expressive by using fixed weights
    # that are not all 1. However, since we don't have params for them, we must fix them.
    # A common choice is just 1s, but maybe a pattern helps?
    # Let's stick with 1s for stability, but maybe scale?
    # Actually, if the hidden units are independent, summing them is fine.
    # The flexibility comes from W1 and b1.
    
    W2 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    b2 = 0.0
    
    # Output
    pred = W2 @ h + b2
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        # If something goes wrong (e.g. overflow), return a safe value
        # But ReLU and linear ops on finite inputs should be finite unless inputs are huge
        # x is in [0.1, 6], params are fitted, so should be safe.
        pred = np.zeros_like(x)
        
    return pred

# EVOLVE-BLOCK-END
