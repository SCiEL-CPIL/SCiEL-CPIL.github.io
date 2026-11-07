import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute basis functions
    # Note: x and y are arrays of shape (N,)
    term0 = 1.0
    term1 = x
    term2 = y
    term3 = x**2
    term4 = y**2
    term5 = x * y
    term6 = np.sin(np.pi * x)
    term7 = np.sin(np.pi * y)
    term8 = np.cos(np.pi * x)
    term9 = np.cos(np.pi * y)
    
    # Linear combination
    pred = (p0 * term0 + p1 * term1 + p2 * term2 + p3 * term3 + 
            p4 * term4 + p5 * term5 + p6 * term6 + p7 * term7 + 
            p8 * term8 + p9 * term9)
            
    return pred
