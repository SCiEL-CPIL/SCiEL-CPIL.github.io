import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a 3-hidden-unit Tanh neural network.
    This structure uses all 10 parameters:
    - 3 weights (w1, w2, w3)
    - 3 slopes (v1, v2, v3)
    - 3 biases (b1, b2, b3)
    - 1 global bias (c)
    
    The Tanh activation provides bounded outputs and smooth gradients,
    making it more robust to the BFGS optimizer's initialization
    compared to higher-degree polynomials which may suffer from
    ill-conditioning or parameter explosion.
    """
    # Unpack parameters
    w1, v1, b1 = params[0], params[1], params[2]
    w2, v2, b2 = params[3], params[4], params[5]
    w3, v3, b3 = params[6], params[7], params[8]
    c = params[9]
    
    # Compute hidden layer activations
    # Using tanh for bounded, smooth non-linearity
    h1 = np.tanh(v1 * x + b1)
    h2 = np.tanh(v2 * x + b2)
    h3 = np.tanh(v3 * x + b3)
    
    # Linear combination of activations + global bias
    return w1 * h1 + w2 * h2 + w3 * h3 + c
