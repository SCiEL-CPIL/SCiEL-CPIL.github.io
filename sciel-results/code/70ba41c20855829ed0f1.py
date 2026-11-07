import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Hybrid model: tanh hidden unit + selected polynomial features
    # Params: w1,w2,w3,b1 (hidden layer); w4 (hidden->output); w5,w6,w7,w8,w9 (direct features)
    
    w1, w2, w3, b1, w4, w5, w6, w7, w8, w9 = params
    
    # Hidden unit with tanh activation - provides nonlinearity
    hidden = np.tanh(w1 * x + w2 * y + w3 * z + b1)
    
    # Output: nonlinear component from hidden unit + linear and interaction terms
    pred = (w4 * hidden + 
            w5 * x + w6 * y + w7 * z + 
            w8 * x * y + w9 * x * z)
    
    return pred
