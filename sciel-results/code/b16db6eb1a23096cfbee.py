import numpy as np

# Fixed constants for 20 hidden units
# We need 20 weights and 20 biases.
# Let's define them as global constants or inside the function.
# To be safe and clean, define inside or as defaults.

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # params shape (10,)
    # Define 20 fixed (w, b) pairs
    # w: 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, -1, -2, -3, -4, -5, -1, -2, -3, -4, -5
    # b: -3, -2, -1, 0, 1, 2, 3, -1.5, 1.5, 0.5, -3, -2, -1, 0, 1, 2, 3, -1.5, 1.5, 0.5
    # This ensures coverage of positive and negative slopes and various intercepts.
    
    w_fixed = np.array([
        1.0, 2.0, 3.0, 4.0, 5.0,
        1.0, 2.0, 3.0, 4.0, 5.0,
        -1.0, -2.0, -3.0, -4.0, -5.0,
        -1.0, -2.0, -3.0, -4.0, -5.0
    ])
    
    b_fixed = np.array([
        -3.0, -2.0, -1.0, 0.0, 1.0,
        2.0, 3.0, -1.5, 1.5, 0.5,
        -3.0, -2.0, -1.0, 0.0, 1.0,
        2.0, 3.0, -1.5, 1.5, 0.5
    ])
    
    # Compute hidden layer activations: shape (20, N)
    # x is (N,)
    # w_fixed is (20,)
    # b_fixed is (20,)
    z = w_fixed[:, np.newaxis] * x[np.newaxis, :] + b_fixed[:, np.newaxis]
    h = np.maximum(0, z) # Shape (20, N)
    
    # Pair them up: 
    # h[0], h[1] weighted by params[0]
    # h[2], h[3] weighted by params[1]
    # ...
    # h[18], h[19] weighted by params[9]
    
    # Reshape h to (10, 2, N)
    h_pairs = h.reshape(10, 2, -1)
    
    # Sum the pairs: (10, N)
    h_sum = np.sum(h_pairs, axis=1)
    
    # Weight by params: params is (10,)
    # y = params @ h_sum
    y = params @ h_sum
    
    return y
