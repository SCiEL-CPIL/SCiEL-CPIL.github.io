"""
Symbolic regression using a small neural network with tanh activations.
The model is:
f(x) = w0 + sum_{i=1}^{3} wi * tanh(vi * (x - mu) / sigma + bi)

Parameters (10 total):
0: w0 (bias)
1: w1, 2: v1, 3: b1
4: w2, 5: v2, 6: b2
7: w3, 8: v3, 9: b3

mu = 3.05 (midpoint of [0.1, 6])
sigma = 1.73 (std of uniform [0.1, 6])
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Small neural network regressor with 3 hidden tanh units.
    Uses centered and scaled input to improve BFGS convergence.
    """
    # Unpack parameters
    w0 = params[0]
    w1, v1, b1 = params[1], params[2], params[3]
    w2, v2, b2 = params[4], params[5], params[6]
    w3, v3, b3 = params[7], params[8], params[9]
    
    # Normalize input x
    # x is in [0.1, 6]
    # mu = 3.05, sigma = (6 - 0.1) / sqrt(12) ~ 1.719
    mu = 3.05
    sigma = 1.719
    z = (x - mu) / sigma
    
    # Compute hidden layers
    h1 = np.tanh(v1 * z + b1)
    h2 = np.tanh(v2 * z + b2)
    h3 = np.tanh(v3 * z + b3)
    
    # Output layer
    y = w0 + w1 * h1 + w2 * h2 + w3 * h3
    
    return y
# EVOLVE-BLOCK-END
