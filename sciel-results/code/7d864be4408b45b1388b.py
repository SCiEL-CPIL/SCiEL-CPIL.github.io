import numpy as np
import time

def equation(x, y, params):
    """
    Symbolic regression with 2 inputs using a rich basis of 10 functions.
    The evaluator fits params via BFGS to minimize training MSE.
    We design a basis that spans a wide variety of functional forms.
    """
    # Ensure inputs are 1D arrays
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    
    n = len(x)
    if n == 0:
        return np.array([])
    
    # Build a diverse set of 10 basis functions
    # We want to cover:
    # - Polynomials up to certain degrees
    # - Trigonometric functions (for periodic behavior)
    # - Rational-like functions (using tanh, etc.)
    # - Interactions
    
    # Polynomial terms
    x2 = x * x
    y2 = y * y
    x3 = x2 * x
    y3 = y2 * y
    
    # Basis functions
    # 1. Constant
    b0 = np.ones_like(x)
    # 2. Linear in x
    b1 = x
    # 3. Linear in y
    b2 = y
    # 4. Quadratic in x
    b3 = x2
    # 5. Quadratic in y
    b4 = y2
    # 6. Interaction x*y
    b5 = x * y
    # 7. Cubic in x
    b6 = x3
    # 8. Cubic in y
    b7 = y3
    # 9. Sine of x (for periodic components)
    b8 = np.sin(x)
    # 10. Cosine of y (for periodic components)
    b9 = np.cos(y)
    
    # Combine
    pred = (params[0] * b0 + params[1] * b1 + params[2] * b2 + 
            params[3] * b3 + params[4] * b4 + params[5] * b5 + 
            params[6] * b6 + params[7] * b7 + params[8] * b8 + 
            params[9] * b9)
    
    return pred
