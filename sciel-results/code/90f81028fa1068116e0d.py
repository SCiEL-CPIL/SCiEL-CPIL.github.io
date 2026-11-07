"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a nonlinear feedforward network with two hidden layers.
    
    The 10 parameters are the final linear output weights. The internal nonlinearity
    (tanh) is fixed, allowing the model to learn complex, non-separable functional
    forms through the fixed internal architecture.
    
    Architecture:
    - Input: (x, y)
    - Hidden layer 1: 5 tanh units with fixed random weights
    - Hidden layer 2: 5 tanh units with fixed random weights
    - Output: linear combination of hidden layer 2 activations with 10 learnable weights
    """
    # Start timer for safety
    t_start = time.time()
    TIME_LIMIT = 140.0  # Safety margin before 150s kill
    
    # Fixed random weights for hidden layers (deterministic seed)
    np.random.seed(42)
    
    # Input dimension: 2 (x, y)
    # Hidden layer 1: 5 units
    W1 = np.random.randn(5, 2) * 1.0
    b1 = np.random.randn(5) * 0.5
    
    # Hidden layer 2: 5 units
    W2 = np.random.randn(5, 5) * 1.0
    b2 = np.random.randn(5) * 0.5
    
    # Build input matrix: each row is [x_i, y_i]
    # x and y are 1D arrays of length n
    inputs = np.column_stack((x, y))  # shape: (n, 2)
    
    # Hidden layer 1: tanh(inputs @ W1.T + b1)
    h1 = np.tanh(inputs @ W1.T + b1)  # shape: (n, 5)
    
    # Hidden layer 2: tanh(h1 @ W2.T + b2)
    h2 = np.tanh(h1 @ W2.T + b2)  # shape: (n, 5)
    
    # Add bias unit to h2 for the linear output
    # We need 10 parameters total. Let's use 5 for h2 activations and 5 for a second set
    # Actually, let's use a different approach: use 10 hidden units total
    # Let's redo with 10 hidden units in a single layer for simplicity
    
    # Reset: Use 10 hidden units with tanh, then linear output with 10 weights
    # But we only have 10 params, so let's make the output layer have 10 weights
    # and the hidden layer produce 10 activations
    
    # Better design: 
    # - 10 hidden units with tanh
    # - Output: sum of params[i] * hidden[i]
    
    # Regenerate weights for 10 hidden units
    np.random.seed(42)
    W_h = np.random.randn(10, 2) * 1.5
    b_h = np.random.randn(10) * 0.5
    
    # Compute hidden activations
    hidden = np.tanh(inputs @ W_h.T + b_h)  # shape: (n, 10)
    
    # Linear output: hidden @ params
    prediction = hidden @ params
    
    # Check time and return
    elapsed = time.time() - t_start
    if elapsed > TIME_LIMIT:
        # Return best effort (which is just the current prediction)
        return prediction
    
    return prediction
# EVOLVE-BLOCK-END
