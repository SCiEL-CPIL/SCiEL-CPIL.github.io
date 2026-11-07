"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fixed-architecture neural network with tanh activations.
    
    Architecture:
    - Input: x, y (2 features)
    - Hidden Layer 1: 5 neurons, tanh activation
    - Hidden Layer 2: 5 neurons, tanh activation
    - Output: 1 neuron, linear activation (weighted sum of hidden layer 2 + bias)
    
    The 10 parameters are the weights connecting Hidden Layer 2 to the output,
    plus a bias term. This makes the function linear in the parameters, 
    ensuring BFGS can optimize them efficiently without getting stuck in 
    local minima related to the weights, while the fixed internal weights 
    provide non-linear feature extraction.
    
    Fixed internal weights are initialized to small random values (deterministic 
    seed for reproducibility) to ensure non-degenerate features.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Ensure x and y are 1D arrays of the same length
    n = x.size
    x_flat = x.flatten()
    y_flat = y.flatten()
    
    # Create input feature matrix [1, x, y]
    # We include a bias term in the inputs for the hidden layers
    X = np.column_stack((np.ones(n), x_flat, y_flat))
    
    # Fixed internal weights (deterministic initialization)
    # Hidden Layer 1: 3 inputs -> 5 neurons
    W1 = np.array([
        [0.1, -0.2, 0.3, 0.1, -0.1],
        [-0.2, 0.1, -0.1, 0.2, 0.1],
        [0.3, 0.2, 0.1, -0.1, 0.2]
    ])
    b1 = np.zeros(5)
    
    # Hidden Layer 2: 5 inputs -> 5 neurons
    W2 = np.array([
        [0.1, 0.2, -0.1, 0.3, -0.2],
        [-0.1, 0.1, 0.2, -0.1, 0.1],
        [0.2, -0.1, 0.1, 0.1, 0.2],
        [0.1, 0.3, -0.2, 0.1, -0.1],
        [-0.2, 0.1, 0.1, -0.2, 0.1]
    ])
    b2 = np.zeros(5)
    
    # Forward pass
    h1 = np.tanh(X @ W1 + b1)  # (n, 5)
    h2 = np.tanh(h1 @ W2 + b2)  # (n, 5)
    
    # Output layer: 5 inputs -> 1 output
    # params[0:5] are weights for h2, params[5] is bias
    # We have 10 params total. Let's use 5 for weights and 1 for bias, 
    # but we need to use all 10. 
    # Let's expand the output layer to use more params or adjust architecture.
    # Alternative: Use 10 params as weights for a 10-dim hidden layer?
    # Let's stick to the directive: "10 parameters are the weights of the final output layer".
    # This implies the hidden layer should have 10 neurons? Or 9 weights + 1 bias?
    # Let's make Hidden Layer 2 have 9 neurons, so we have 9 weights + 1 bias = 10 params.
    
    # Re-initialize fixed weights for 9-neuron hidden layer 2
    # Hidden Layer 1: 3 inputs -> 5 neurons (unchanged)
    # Hidden Layer 2: 5 inputs -> 9 neurons
    W2_new = np.array([
        [0.1, 0.2, -0.1, 0.3, -0.2, 0.1, 0.2, -0.1, 0.3],
        [-0.1, 0.1, 0.2, -0.1, 0.1, 0.2, -0.1, 0.3, -0.2],
        [0.2, -0.1, 0.1, 0.1, 0.2, -0.1, 0.2, 0.1, -0.1],
        [0.1, 0.3, -0.2, 0.1, -0.1, 0.2, 0.1, -0.2, 0.1],
        [-0.2, 0.1, 0.1, -0.2, 0.1, 0.1, -0.1, 0.2, -0.2]
    ])
    b2_new = np.zeros(9)
    
    # Recompute forward pass with 9-neuron hidden layer 2
    h1 = np.tanh(X @ W1 + b1)  # (n, 5)
    h2 = np.tanh(h1 @ W2_new + b2_new)  # (n, 9)
    
    # Output: 9 weights + 1 bias
    # params[0:9] are weights for h2, params[9] is bias
    if params.size >= 10:
        w_out = params[:9]
        b_out = params[9]
    else:
        # Fallback if params is shorter (should not happen per contract)
        w_out = np.ones(9)
        b_out = 1.0
        
    output = h2 @ w_out + b_out
    
    return output
# EVOLVE-BLOCK-END
