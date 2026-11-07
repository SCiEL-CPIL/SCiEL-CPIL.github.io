import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: Full Polynomial Basis of Degree 3.
    This spans the space of all polynomials up to degree 3 in x and y.
    
    Basis Functions:
    0: 1
    1: x
    2: y
    3: x^2
    4: xy
    5: y^2
    6: x^3
    7: x^2 * y
    8: x * y^2
    9: y^3
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Compute basis functions
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = x * x
    b4 = x * y
    b5 = y * y
    b6 = x * x * x
    b7 = x * x * y
    b8 = x * y * y
    b9 = y * y * y
    
    # Linear combination
    pred = (p[0] * b0 + 
            p[1] * b1 + 
            p[2] * b2 + 
            p[3] * b3 + 
            p[4] * b4 + 
            p[5] * b5 + 
            p[6] * b6 + 
            p[7] * b7 + 
            p[8] * b8 + 
            p[9] * b9)
            
    # Safety check
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
        
    return pred
