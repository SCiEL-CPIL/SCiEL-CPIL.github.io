import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a flexible sigmoid-based unit plus polynomial terms.
    
    The model uses a learnable sigmoid (tanh) unit to capture non-linear smooth behavior:
    y(x) = c0 + c1*x + c2*x^2 + c3*x^3 + 
           c4 * tanh(c5 * (x - c6)) + 
           c7 * tanh(c8 * (x - c9))
    
    This allows the function to have flexible inflection points and asymptotic plateaus,
    which are common in smooth functions, while the polynomial part handles global trends.
    The parameters are:
    c0: constant
    c1: linear coeff
    c2: quadratic coeff
    c3: cubic coeff
    c4: amplitude of first sigmoid
    c5: slope of first sigmoid
    c6: center of first sigmoid
    c7: amplitude of second sigmoid
    c8: slope of second sigmoid
    c9: center of second sigmoid
    """
    c0 = params[0]
    c1 = params[1]
    c2 = params[2]
    c3 = params[3]
    c4 = params[4]
    c5 = params[5]
    c6 = params[6]
    c7 = params[7]
    c8 = params[8]
    c9 = params[9]
    
    # Compute polynomial part
    x2 = x * x
    poly = c0 + c1 * x + c2 * x2 + c3 * x2 * x
    
    # Compute sigmoid parts
    # Using tanh for bounded smooth nonlinearity
    sig1 = c4 * np.tanh(c5 * (x - c6))
    sig2 = c7 * np.tanh(c8 * (x - c9))
    
    y = poly + sig1 + sig2
    
    return y
