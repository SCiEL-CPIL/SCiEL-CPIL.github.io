"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    2-layer Neural Network (MLP) with 3 hidden neurons (ReLU) and 1 output neuron (Linear).
    Parameters:
    p[0:3] -> Weights W1 (3x1)
    p[3:6] -> Biases b1 (3)
    p[6:9] -> Weights W2 (3)
    p[9]   -> Bias b2 (1)
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Normalize input to [0, 1] range approximately to keep activations stable
    # Domain is [0.1, 6]
    x_norm = (x - 0.1) / 5.9
    
    # Reshape params
    W1 = params[0:3]  # (3,)
    b1 = params[3:6]  # (3,)
    W2 = params[6:9]  # (3,)
    b2 = params[9]    # scalar
    
    # Hidden layer: 3 neurons
    # z = x_norm * W1 + b1
    # W1 is (3,), x_norm is (N,). We need (N, 3).
    # z[i, j] = x_norm[i] * W1[j] + b1[j]
    z = np.outer(x_norm, W1) + b1  # Shape (N, 3)
    
    # Activation: ReLU
    h = np.maximum(0, z)
    
    # Output layer: Linear
    # y = h @ W2 + b2
    y = h @ W2 + b2
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=1e10, neginf=-1e10)
        
    return y

# EVOLVE-BLOCK-END
