import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Hidden layer 1
    z1 = p0 * x + p1 * y + p2
    h1 = np.tanh(z1)
    
    # Hidden layer 2
    z2 = p3 * x + p4 * y + p5
    h2 = np.tanh(z2)
    
    # Output layer
    # p6 is weight for h1, p7 is weight for h2, p8 is bias, p9 is global scale
    out = p9 * (p6 * h1 + p7 * h2 + p8)
    
    return out
