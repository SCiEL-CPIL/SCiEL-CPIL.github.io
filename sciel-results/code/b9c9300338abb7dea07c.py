import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.flatten()
        
    p = np.asarray(params, dtype=np.float64)
    
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)
        
    # Strategy: Product of two low-degree polynomials P3(x) * Q6(x)
    # P3(x) = p[0] + p[1]*x + p[2]*x^2 + p[3]*x^3 (4 coeffs)
    # Q6(x) = p[4] + p[5]*x + p[6]*x^2 + p[7]*x^3 + p[8]*x^4 + p[9]*x^5 (6 coeffs)
    # Total 10 coeffs. Product is degree 9.
    # This creates a non-convex but structured landscape that BFGS can navigate.
    
    # Compute powers of x
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    x6 = x3 * x3
    x7 = x6 * x
    x8 = x4 * x4
    x9 = x8 * x
    
    # P3(x) = p[0] + p[1]*x + p[2]*x^2 + p[3]*x^3
    P = p[0] + p[1]*x + p[2]*x2 + p[3]*x3
    
    # Q6(x) = p[4] + p[5]*x + p[6]*x^2 + p[7]*x^3 + p[8]*x^4 + p[9]*x^5
    Q = p[4] + p[5]*x + p[6]*x2 + p[7]*x3 + p[8]*x4 + p[9]*x5
    
    y = P * Q
    
    # Ensure finite outputs
    y = np.where(np.isfinite(y), y, 0.0)
    
    return y
