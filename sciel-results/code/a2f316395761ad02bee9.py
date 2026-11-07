import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Standardize inputs to help optimization
    # Mean of [0.5, 2] is 1.25, Std is approx 0.433
    mean = 1.25
    std = 0.433
    
    X = (x - mean) / std
    Y = (y - mean) / std
    Z = (z - mean) / std
    
    # Construct basis
    # 1: Bias
    # 2: X
    # 3: Y
    # 4: Z
    # 5: X^2
    # 6: Y^2
    # 7: Z^2
    # 8: XY
    # 9: XZ
    # 10: YZ
    
    # Ensure params is 1D
    p = params.flatten()
    
    # Check length just in case, though contract says 10
    if len(p) < 10:
         # Fallback to linear if params are short (should not happen)
         return p[0]*x + p[1]*y + p[2]*z + p[3]
         
    term1 = np.ones_like(x)
    term2 = X
    term3 = Y
    term4 = Z
    term5 = X * X
    term6 = Y * Y
    term7 = Z * Z
    term8 = X * Y
    term9 = X * Z
    term10 = Y * Z
    
    return (p[0] * term1 + 
            p[1] * term2 + 
            p[2] * term3 + 
            p[3] * term4 + 
            p[4] * term5 + 
            p[5] * term6 + 
            p[6] * term7 + 
            p[7] * term8 + 
            p[8] * term9 + 
            p[9] * term10)
