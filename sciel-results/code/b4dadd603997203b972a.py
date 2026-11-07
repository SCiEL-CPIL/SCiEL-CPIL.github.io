import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with a mixed basis covering polynomial, trigonometric,
    exponential, rational, and absolute-value terms. This gives BFGS many more
    degrees of freedom to fit diverse hidden functions.
    
    Basis (10 terms):
    0: 1
    1: x
    2: y
    3: x^2
    4: y^2
    5: sin(x)
    6: cos(y)
    7: exp(-x^2 - y^2)  (Gaussian bump)
    8: x * y
    9: abs(x - y)
    """
    # Polynomial terms
    term0 = np.ones_like(x)
    term1 = x
    term2 = y
    term3 = x * x
    term4 = y * y
    
    # Trigonometric terms
    term5 = np.sin(x)
    term6 = np.cos(y)
    
    # Exponential (Gaussian) term
    term7 = np.exp(-(x * x + y * y))
    
    # Interaction term
    term8 = x * y
    
    # Absolute value term
    term9 = np.abs(x - y)
    
    # Combine all terms
    phi = (params[0] * term0 +
           params[1] * term1 +
           params[2] * term2 +
           params[3] * term3 +
           params[4] * term4 +
           params[5] * term5 +
           params[6] * term6 +
           params[7] * term7 +
           params[8] * term8 +
           params[9] * term9)
    
    return phi
