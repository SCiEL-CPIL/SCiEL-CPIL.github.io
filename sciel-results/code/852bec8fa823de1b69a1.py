import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a hybrid basis of polynomial and smooth nonlinear terms.
    
    y(x) = c0 + c1*x + c2*x^2 + c3*x^3 + c4*x^4 + 
           c5*tanh(x) + c6*exp(-x^2) + c7/(1+x^2) + c8*sin(x)*exp(-0.5*x^2) + c9*cos(x)*exp(-0.5*x^2)
    
    This hybrid basis covers polynomials, smooth nonlinearities, and damped oscillations.
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
    
    x2 = x * x
    exp_neg_x2 = np.exp(-x2)
    
    y = c0 + c1 * x + c2 * x2 + c3 * x2 * x + c4 * x2 * x2 + \
        c5 * np.tanh(x) + \
        c6 * exp_neg_x2 + \
        c7 / (1.0 + x2) + \
        c8 * np.sin(x) * exp_neg_x2 + \
        c9 * np.cos(x) * exp_neg_x2
        
    return y
