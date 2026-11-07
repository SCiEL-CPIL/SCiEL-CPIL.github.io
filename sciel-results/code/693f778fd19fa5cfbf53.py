"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Two-hidden-layer tanh neural network.
    The 10 params are weights for 9 hidden layer 2 neurons and 1 constant bias.
    Hidden layers have fixed random weights initialized to provide diverse basis functions
    with O(1) magnitude to ensure stable optimization.
    """
    # Fixed random weights for hidden layers
    # Using a different seed and scaling to ensure features are O(1)
    rng = np.random.RandomState(99)
    
    n_hidden1 = 32
    n_hidden2 = 32
    
    # Hidden layer 1: input (x, y) -> 32 neurons
    # Scale weights to 1.0 to ensure non-linearity and O(1) activations
    W1 = rng.randn(n_hidden1, 2) * 1.0 
    b1 = rng.rand(n_hidden1) - 0.5
    
    # Hidden layer 2: 32 neurons -> 32 neurons
    W2 = rng.randn(n_hidden2, n_hidden1) * 1.0
    b2 = rng.rand(n_hidden2) - 0.5
    
    # Compute hidden layer activations
    # x and y are 1D arrays of length n
    # Reshape for broadcasting
    X = x[:, None]
    Y = y[:, None]
    
    z1 = X * W1[:, 0] + Y * W1[:, 1] + b1
    h1 = np.tanh(z1)  # shape: (n, 32)
    
    z2 = h1 @ W2.T + b2
    h2 = np.tanh(z2)  # shape: (n, 32)
    
    # Output: linear combination of 9 selected h2 features + 1 constant
    # params[0:9] -> weights for h2[:, 0] ... h2[:, 8]
    # params[9]   -> constant bias
    
    out = (params[0] * h2[:, 0] + 
           params[1] * h2[:, 1] + 
           params[2] * h2[:, 2] + 
           params[3] * h2[:, 3] + 
           params[4] * h2[:, 4] + 
           params[5] * h2[:, 5] + 
           params[6] * h2[:, 6] + 
           params[7] * h2[:, 7] + 
           params[8] * h2[:, 8] +
           params[9])
    
    return out
# EVOLVE-BLOCK-END
