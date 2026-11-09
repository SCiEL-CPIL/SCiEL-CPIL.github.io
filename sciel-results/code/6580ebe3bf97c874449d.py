import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Neural residual basis for nonlinear oscillator acceleration.
    
    Architecture:
    - Normalize inputs to ~[-1, 1] using known training ranges
    - Fixed hidden layer: 7 tanh units with a fixed weight matrix
    - Output: linear combination of hidden features + skip connections
    - 10 total parameters: 7 output weights for hidden units + 3 skip weights
    
    This provides a universal approximator structure with good conditioning.
    """
    # Normalize inputs to approximately [-1, 1]
    # t: [30, 50] -> center=40, half_range=10
    # x: [-0.2056, 0.1749] -> center=-0.01535, half_range=0.19025
    # v: [-0.3391, 0.3624] -> center=0.01165, half_range=0.35075
    t_norm = (t - 40.0) / 10.0
    x_norm = (x + 0.01535) / 0.19025
    v_norm = (v - 0.01165) / 0.35075
    
    # Stack normalized inputs: shape (n, 3)
    inputs = np.stack([t_norm, x_norm, v_norm], axis=-1)
    
    # Fixed weight matrix for hidden layer: (3, 7)
    # Use a deterministic pseudo-random matrix with good spread
    # These are FIXED, not learned
    W = np.array([
        [ 1.0, -0.7,  0.5,  0.9, -1.1,  0.3, -0.8],
        [-0.6,  1.2, -0.4, -0.8,  0.7,  1.0, -0.5],
        [ 0.4, -0.9,  1.1, -0.3,  0.6, -0.7,  0.9]
    ])
    
    # Hidden layer: tanh(inputs @ W) -> shape (n, 7)
    hidden = np.tanh(inputs @ W)
    
    # Output weights: first 7 for hidden units, last 3 for skip connections
    h_weights = params[:7]
    skip_weights = params[7:10]
    
    # Linear combination of hidden features
    h_out = hidden @ h_weights  # shape (n,)
    
    # Skip connections: linear from normalized inputs
    skip_out = inputs @ skip_weights  # shape (n,)
    
    return h_out + skip_out
