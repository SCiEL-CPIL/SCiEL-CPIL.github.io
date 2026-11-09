import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Ensure inputs are 1D
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Basis functions for full cubic polynomial in 2 variables
    # Order:
    # 0: 1
    # 1: x
    # 2: y
    # 3: x^2
    # 4: xy
    # 5: y^2
    # 6: x^3
    # 7: x^2y
    # 8: xy^2
    # 9: y^3
    
    term0 = np.ones_like(x)
    term1 = x
    term2 = y
    term3 = x**2
    term4 = x * y
    term5 = y**2
    term6 = x**3
    term7 = x**2 * y
    term8 = x * y**2
    term9 = y**3
    
    return (p0 * term0 + p1 * term1 + p2 * term2 + 
            p3 * term3 + p4 * term4 + p5 * term5 + 
            p6 * term6 + p7 * term7 + p8 * term8 + 
            p9 * term9)
