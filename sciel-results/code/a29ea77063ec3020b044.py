# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

# Precompute fixed network components to avoid recomputation and ensure speed
# Network: 1 input -> 5 hidden (tanh) -> 1 output
# params[0:5] are weights from input to hidden layer
# params[5:10] are weights from hidden layer to output
# Biases are fixed to 0 for simplicity, or could be derived from params if needed, 
# but with only 10 params, we must be efficient. 
# Actually, standard NN needs biases. 
# Let's structure:
# h = tanh(W_in * x + b_in)
# y = W_out * h + b_out
# We have 10 params. 
# If we use 5 hidden nodes:
# W_in: 1x5 (5 params)
# b_in: 1x5 (5 params) -> Total 10. No W_out or b_out. This is a linear readout from hidden features.
# This is equivalent to a linear model with 5 tanh(x) basis functions.
# Let's try a slightly more expressive fixed structure if possible, or stick to this robust "adaptive basis" approach.
# The prompt says "closed-form expression". A linear combination of tanh(x) is closed form.
# This is structurally different from a linear polynomial in x.

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Equation using a fixed 5-node tanh hidden layer with learned weights and biases.
    params[0:5] = weights for input to hidden
    params[5:10] = biases for hidden layer
    Output is the sum of the hidden activations (implied weight 1) or we can use a fixed output weight.
    To make it a proper regression, let's assume the output is the sum of the hidden units.
    y = sum(tanh(w_i * x + b_i))
    This is a universal approximator with fixed output weights of 1.
    """
    # Ensure x is 1D for safe operations
    x = np.asarray(x, dtype=np.float64).flatten()
    
    # Extract parameters
    w_in = params[0:5]
    b_in = params[5:10]
    
    # Compute hidden layer activations
    # x is shape (N,), w_in is (5,), b_in is (5,)
    # We need to broadcast x to (N, 1) and w_in to (1, 5)
    x_expanded = x[:, np.newaxis]  # (N, 1)
    
    # z = x * w_in + b_in
    # (N, 1) * (5,) -> (N, 5)
    # (N, 5) + (5,) -> (N, 5)
    z = x_expanded * w_in[np.newaxis, :] + b_in[np.newaxis, :]
    
    # h = tanh(z)
    h = np.tanh(z)  # (N, 5)
    
    # Output: sum of hidden activations
    # This assumes the final layer has all weights equal to 1.
    # This is a valid closed-form expression.
    y_pred = np.sum(h, axis=1)
    
    return y_pred
# EVOLVE-BLOCK-END
