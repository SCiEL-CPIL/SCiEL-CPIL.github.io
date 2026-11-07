import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a 10-term basis combining 
    Chebyshev polynomials and specific cross terms to capture 
    interactions and higher-order effects more effectively than 
    pure monomials or low-degree Chebyshev products.
    
    Basis terms:
    0: 1
    1: x
    2: y
    3: x^2 - 0.5 (centered quadratic x)
    4: y^2 - 0.5 (centered quadratic y)
    5: x^3 - 1.5x (centered cubic x, related to T3)
    6: y^3 - 1.5y (centered cubic y, related to T3)
    7: xy
    8: x^2y
    9: xy^2
    
    This basis is designed to be well-conditioned for BFGS starting from ones,
    and includes interaction terms (xy, x^2y, xy^2) which are often critical
    in 2D functions that pure univariate sums miss.
    """
    start_time = time.time()
    
    try:
        # Precompute powers and centered terms for stability
        x2 = x * x
        y2 = y * y
        x3 = x2 * x
        y3 = y2 * y
        
        # Basis terms
        # Note: We use centered versions for higher orders to help optimization
        # and reduce correlation between parameters.
        
        b0 = np.ones_like(x)
        b1 = x
        b2 = y
        b3 = x2 - 0.5
        b4 = y2 - 0.5
        b5 = x3 - 1.5 * x
        b6 = y3 - 1.5 * y
        b7 = x * y
        b8 = x2 * y
        b9 = x * y2
        
        prediction = (params[0] * b0 + 
                      params[1] * b1 + 
                      params[2] * b2 + 
                      params[3] * b3 + 
                      params[4] * b4 + 
                      params[5] * b5 + 
                      params[6] * b6 + 
                      params[7] * b7 + 
                      params[8] * b8 + 
                      params[9] * b9)
        
        if not np.all(np.isfinite(prediction)):
            return np.zeros_like(x)
            
        return prediction
        
    except Exception:
        return np.zeros_like(x)
