"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression candidate using a small Multi-Layer Perceptron (MLP).
    
    Architecture:
    - Input: 2 features (x, y)
    - Hidden Layer: 2 neurons with tanh activation
    - Output Layer: 1 neuron with linear activation
    - Global Scale: 1 parameter to adjust output magnitude
    
    Parameters (10 total):
    p0: weight x -> h1
    p1: weight y -> h1
    p2: bias h1
    p3: weight x -> h2
    p4: weight y -> h2
    p5: bias h2
    p6: weight h1 -> out
    p7: weight h2 -> out
    p8: bias out
    p9: global scale factor
    
    This non-linear parametric model can approximate a wider class of functions
    than a simple linear basis of polynomials and trig functions, especially
    functions with sharp transitions or complex interactions.
    """
    # Ensure x and y are 1D arrays
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    
    # Extract parameters
    w1x, w1y, b1 = params[0], params[1], params[2]
    w2x, w2y, b2 = params[3], params[4], params[5]
    wo1, wo2, bo = params[6], params[7], params[8]
    scale = params[9]
    
    # Compute hidden layer activations
    z1 = w1x * x + w1y * y + b1
    z2 = w2x * x + w2y * y + b2
    
    h1 = np.tanh(z1)
    h2 = np.tanh(z2)
    
    # Compute output
    pred = scale * (wo1 * h1 + wo2 * h2 + bo)
    
    return pred
# EVOLVE-BLOCK-END
