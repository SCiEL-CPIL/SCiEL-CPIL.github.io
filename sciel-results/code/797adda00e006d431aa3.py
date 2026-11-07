"""Symbolic regression with 1 input; Hybrid MLP + RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # Safety margin before 150s kill


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model combining a small MLP (1-2-1) with RBF basis functions.
    
    Parameters (length 10):
    - params[0:2]: Weights from input to hidden layer (w1)
    - params[2:4]: Biases for hidden layer (b1)
    - params[4:6]: Weights from hidden to output (w2)
    - params[6]:   Output bias (b2)
    - params[7:10]: RBF coefficients (3 RBF centers)
    
    The input x is scaled to [-1, 1] for numerical stability.
    RBF centers are fixed at 1.5, 3.0, 4.5 with sigma=1.0.
    """
    x = np.asarray(x, dtype=float)
    params = np.asarray(params, dtype=float).flatten()
    
    # Ensure exactly 10 parameters
    if len(params) < 10:
        params = np.concatenate([params, np.ones(10 - len(params))])
    else:
        params = params[:10]
    
    # Scale input x from [0.1, 6] to roughly [-1, 1]
    x_scaled = (x - 3.05) / 3.0
    
    # --- MLP Part (1-2-1) ---
    w1 = params[0:2]
    b1 = params[2:4]
    w2 = params[4:6]
    b2 = params[6]
    
    h1 = np.tanh(x_scaled * w1[0] + b1[0])
    h2 = np.tanh(x_scaled * w1[1] + b1[1])
    
    y_mlp = w2[0] * h1 + w2[1] * h2 + b2
    
    # --- RBF Part ---
    # Fixed centers at 1.5, 3.0, 4.5 with sigma=1.0
    centers = np.array([1.5, 3.0, 4.5])
    sigma = 1.0
    
    # Compute RBF basis: shape (N, 3)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_basis = np.exp(-0.5 * (diff / sigma) ** 2)  # shape (N, 3)
    
    # RBF coefficients
    rbf_coeffs = params[7:10]
    
    y_rbf = rbf_basis @ rbf_coeffs
    
    # Combine
    y = y_mlp + y_rbf
    
    return y
# EVOLVE-BLOCK-END
