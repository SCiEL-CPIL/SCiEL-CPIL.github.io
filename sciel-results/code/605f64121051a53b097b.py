"""Symbolic regression using a small neural network (1 hidden layer)."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a 1-hidden-layer neural network.
    
    Architecture:
    - Input: x (scalar per point)
    - Hidden Layer: 3 neurons with Tanh activation
    - Output Layer: Linear combination of hidden activations + bias
    
    Parameters (length 10):
    - params[0:3]: Weights from input to hidden layer (W1)
    - params[3:6]: Biases for hidden layer (b1)
    - params[6:9]: Weights from hidden layer to output (W2)
    - params[9]:   Bias for output layer (b2)
    
    This non-linear model can capture complex smooth functions better than a 
    fixed linear basis, and BFGS can optimize the non-linear parameters effectively.
    """
    x = np.asarray(x, dtype=np.float64)
    # Reshape x to column vector for matrix multiplication
    # x shape: (N, 1)
    x_col = x[:, np.newaxis]
    
    # Unpack parameters
    W1 = params[0:3]   # (3,)
    b1 = params[3:6]   # (3,)
    W2 = params[6:9]   # (3,)
    b2 = params[9]     # scalar
    
    # Compute hidden layer
    # x_col @ W1.T -> (N, 1) @ (1, 3) -> (N, 3)
    # Add bias b1 (broadcasts to (N, 3))
    z1 = x_col @ W1[np.newaxis, :] + b1[np.newaxis, :]
    
    # Apply activation
    h1 = np.tanh(z1)
    
    # Compute output
    # h1 @ W2 -> (N, 3) @ (3,) -> (N,)
    # Add bias b2
    y = h1 @ W2 + b2
    
    return y
# EVOLVE-BLOCK-END
