import numpy as np
def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    p = params
    
    # Parameters:
    # w0: bias
    # w1, w2: output weights
    # a1, b1, c1: weights for first hidden neuron
    # a2, b2, c2: weights for second hidden neuron
    # d2: bias for second hidden neuron (or we can distribute bias differently)
    
    # Let's use:
    # h1 = tanh(a1*x + b1*y + c1*z)
    # h2 = tanh(a2*x + b2*y + c2*z + d2)
    # out = w0 + w1*h1 + w2*h2
    
    w0 = p[0]
    w1 = p[1]
    w2 = p[2]
    
    a1 = p[3]
    b1 = p[4]
    c1 = p[5]
    
    a2 = p[6]
    b2 = p[7]
    c2 = p[8]
    d2 = p[9]
    
    h1 = np.tanh(a1 * x + b1 * y + c1 * z)
    h2 = np.tanh(a2 * x + b2 * y + c2 * z + d2)
    
    return w0 + w1 * h1 + w2 * h2
