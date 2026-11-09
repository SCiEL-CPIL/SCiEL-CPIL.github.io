import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a basis of hyperbolic tangent functions.
    This basis is chosen to model smooth, bounded, and interacting features
    that may not be well-captured by low-degree polynomials or low-frequency
    Fourier series. The model remains linear in the parameters, ensuring
    robust optimization with BFGS.
    
    Basis terms:
    1. 1 (Constant)
    2. x
    3. y
    4. tanh(x)
    5. tanh(y)
    6. tanh(x + y)
    7. tanh(x - y)
    8. tanh(x^2 - y^2)
    9. tanh(2xy)
    10. tanh(x^2 + y^2)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Compute basis terms
    b1 = np.ones_like(x)
    b2 = x
    b3 = y
    
    # Non-linear terms using tanh
    # Note: np.tanh is numerically stable and smooth
    b4 = np.tanh(x)
    b5 = np.tanh(y)
    
    b6 = np.tanh(x + y)
    b7 = np.tanh(x - y)
    
    x2 = x * x
    y2 = y * y
    b8 = np.tanh(x2 - y2)
    b9 = np.tanh(2.0 * x * y)
    b10 = np.tanh(x2 + y2)
    
    # Linear combination
    pred = (params[0] * b1 + 
            params[1] * b2 + 
            params[2] * b3 + 
            params[3] * b4 + 
            params[4] * b5 + 
            params[5] * b6 + 
            params[6] * b7 + 
            params[7] * b8 + 
            params[8] * b9 + 
            params[9] * b10)
            
    return pred
