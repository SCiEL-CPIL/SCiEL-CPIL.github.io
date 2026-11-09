"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate using a single-layer ReLU neural network.
    The 10 parameters are the output layer weights and bias.
    The hidden layer uses fixed centers and widths for RBF-like basis functions,
    making the model linear in parameters (convex loss for BFGS).
    
    Hidden units:
    - 4 RBFs on normalized b, s, t, p (Gaussian)
    - 4 ReLUs on linear combinations (to capture non-linear interactions)
    - 1 bias
    Total: 9 hidden units + 1 bias = 10 params
    """
    # Normalize inputs to roughly [-1, 1] or similar stable range
    b_norm = (b - 3.0) / 2.0
    s_norm = (s - 20.0) / 10.0
    t_norm = (temp - 20.0) / 10.0
    p_norm = (pH - 7.0) / 3.0
    
    # Unpack params: 9 weights for hidden units + 1 bias
    w1, w2, w3, w4, w5, w6, w7, w8, w9, bias = params
    
    # Hidden layer activations
    # 4 Gaussian RBFs centered at 0 with width 1.0
    h1 = np.exp(-(b_norm ** 2) / 2.0)
    h2 = np.exp(-(s_norm ** 2) / 2.0)
    h3 = np.exp(-(t_norm ** 2) / 2.0)
    h4 = np.exp(-(p_norm ** 2) / 2.0)
    
    # 4 ReLU units on linear combinations (interactions)
    # These capture non-linear interactions
    h5 = np.maximum(0.0, b_norm + s_norm)
    h6 = np.maximum(0.0, b_norm - s_norm)
    h7 = np.maximum(0.0, t_norm + p_norm)
    h8 = np.maximum(0.0, t_norm - p_norm)
    
    # One more ReLU for overall interaction
    h9 = np.maximum(0.0, b_norm * s_norm + t_norm * p_norm)
    
    # Output layer: linear combination
    y = (bias + 
         w1 * h1 + 
         w2 * h2 + 
         w3 * h3 + 
         w4 * h4 + 
         w5 * h5 + 
         w6 * h6 + 
         w7 * h7 + 
         w8 * h8 + 
         w9 * h9)
         
    return y

# EVOLVE-BLOCK-END
