import numpy as np

def equation(x, y, params):
    """
    Non-linear model using a small neural network structure (2 hidden units with tanh)
    plus a global linear trend term.
    
    Model:
    y = v1 * tanh(w1x*x + w1y*y + b1) 
      + v2 * tanh(w2x*x + w2y*y + b2) 
      + v3 * (x + y) 
      + b_out
    
    Parameters (length 10):
    [w1x, w1y, b1, w2x, w2y, b2, v1, v2, v3, b_out]
    """
    # Unpack parameters
    w1x = params[0]
    w1y = params[1]
    b1 = params[2]
    w2x = params[3]
    w2y = params[4]
    b2 = params[5]
    v1 = params[6]
    v2 = params[7]
    v3 = params[8]
    b_out = params[9]
    
    # Compute hidden layers
    h1 = np.tanh(w1x * x + w1y * y + b1)
    h2 = np.tanh(w2x * x + w2y * y + b2)
    
    # Combine
    return v1 * h1 + v2 * h2 + v3 * (x + y) + b_out
