# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Single-hidden-layer neural network with ReLU activations.
    
    Architecture:
    - Input: x (scalar)
    - Hidden layer: 5 ReLU units
    - Output: linear combination of hidden activations + bias
    
    Parameters (10 total):
    - params[0:5]: weights from input to hidden layer (w1_i)
    - params[5:10]: biases for hidden layer (b_i)
    
    Wait, that's only 10 params for weights and biases. We need output weights too.
    Let me restructure:
    - 5 hidden units with fixed centers and widths, or use all 10 params differently.
    
    Better approach with 10 params:
    - 4 hidden ReLU units
    - params[0:4]: hidden weights (w_i)  
    - params[4:8]: hidden biases (b_i)
    - params[8:10]: output weights (v_i) for first 2 hidden units + bias
    
    Actually, let's use a simpler structure:
    - 5 ReLU units with shared output weight structure
    
    Cleanest: Use 5 ReLU units where:
    - params[0:5] = hidden layer weights (w_i)
    - params[5:10] = hidden layer biases (b_i)
    - Output = sum of ReLU(w_i * x + b_i) with fixed equal output weights
    
    This gives a piecewise linear function with 5 "kinks" that BFGS can fit.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    
    # 5 hidden ReLU units
    # params[0:5] = weights, params[5:10] = biases
    w = params[0:5]
    b = params[5:10]
    
    # Compute hidden activations: ReLU(w_i * x + b_i) for i=0..4
    # Shape: (n, 5)
    Z = np.zeros((n, 5), dtype=np.float64)
    for i in range(5):
        Z[:, i] = np.maximum(0.0, w[i] * x + b[i])
    
    # Output: sum of hidden activations (equal output weights) + implicit bias from ReLU(0)=0
    # This is a piecewise linear function
    pred = Z.sum(axis=1)
    
    return pred
# EVOLVE-BLOCK-END
