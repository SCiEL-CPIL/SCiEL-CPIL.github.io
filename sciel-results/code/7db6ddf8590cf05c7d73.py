"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Two-hidden-layer tanh neural network with 10 learnable output parameters.
    The model is linear in params, making it suitable for BFGS optimization.
    
    Architecture:
    - Input: (x, y)
    - Hidden layer 1: 8 neurons, tanh activation, fixed weights
    - Hidden layer 2: 8 neurons, tanh activation, fixed weights
    - Output: linear combination of h2 features + polynomial basis terms
    """
    # Fixed random weights for hidden layers (deterministic)
    rng = np.random.RandomState(123)
    
    # Hidden layer 1: input (x, y) -> 8 neurons
    W1 = rng.randn(8, 2) * np.sqrt(2.0 / 2)
    b1 = rng.randn(8) * 0.1
    
    # Hidden layer 2: 8 neurons -> 8 neurons
    W2 = rng.randn(8, 8) * np.sqrt(2.0 / 8)
    b2 = rng.randn(8) * 0.1
    
    # Compute hidden layer activations
    z1 = x[:, None] * W1[None, :, 0] + y[:, None] * W1[None, :, 1] + b1[None, :]
    h1 = np.tanh(z1)  # shape: (n, 8)
    
    z2 = h1 @ W2.T + b2[None, :]
    h2 = np.tanh(z2)  # shape: (n, 8)
    
    # Output: linear combination of h2 and polynomial basis
    # params[0:8] weights for h2, params[8] for constant, params[9] for x*y
    out = (h2 * params[0:8][None, :]).sum(axis=1)
    out += params[8]
    out += params[9] * x * y
    
    return out
# EVOLVE-BLOCK-END
