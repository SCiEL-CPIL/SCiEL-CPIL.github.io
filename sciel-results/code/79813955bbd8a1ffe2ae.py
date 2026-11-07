import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # params is expected to be of length 10
    # Basis: 1, x, y, z, x^2, y^2, z^2, xy, xz, yz
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute terms
    # Note: Using raw x, y, z
    term0 = np.ones_like(x)
    term1 = x
    term2 = y
    term3 = z
    term4 = x * x
    term5 = y * y
    term6 = z * z
    term7 = x * y
    term8 = x * z
    term9 = y * z
    
    # Combine
    pred = (p0 * term0 + p1 * term1 + p2 * term2 + p3 * term3 + 
            p4 * term4 + p5 * term5 + p6 * term6 + 
            p7 * term7 + p8 * term8 + p9 * term9)
            
    return pred
