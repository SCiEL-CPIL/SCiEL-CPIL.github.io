import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Compute basis terms
    # Ensure shapes are compatible (they should be 1D arrays of same length)
    
    term1 = np.ones_like(t)
    term2 = x
    term3 = v
    term4 = x**2
    term5 = v**2
    term6 = x * v
    term7 = x**3
    term8 = v**3
    term9 = np.sin(t)
    term10 = np.cos(t)
    
    # Combine with params
    # params is length 10
    return (params[0]*term1 + params[1]*term2 + params[2]*term3 + 
            params[3]*term4 + params[4]*term5 + params[5]*term6 +
            params[6]*term7 + params[7]*term8 + params[8]*term9 +
            params[9]*term10)
