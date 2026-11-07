"""Program using a small MLP structure for symbolic regression."""
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form symbolic regression candidate.
    Uses a 1-hidden-layer neural network (MLP) with 3 hidden units.
    This allows learning non-linear features with learned breakpoints,
    unlike fixed basis functions.
    
    Parameters:
    0-1: w1, b1 (1st hidden unit)
    2-3: w2, b2 (2nd hidden unit)
    4-5: w3, b3 (3rd hidden unit)
    6-8: v1, v2, v3 (output weights)
    9: c (bias)
    """
    # Unpack parameters
    w1, b1 = params[0], params[1]
    w2, b2 = params[2], params[3]
    w3, b3 = params[4], params[5]
    v1, v2, v3 = params[6], params[7], params[8]
    c = params[9]
    
    # Calculate hidden layer activations
    # Using tanh for smooth, zero-centered non-linearity
    h1 = np.tanh(w1 * x + b1)
    h2 = np.tanh(w2 * x + b2)
    h3 = np.tanh(w3 * x + b3)
    
    # Linear combination of hidden units
    y = v1 * h1 + v2 * h2 + v3 * h3 + c
    
    return y
