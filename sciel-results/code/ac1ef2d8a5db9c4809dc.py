import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # params[0]: weight for tanh 1
    # params[1]: bias for tanh 1
    # params[2]: weight for tanh 2
    # params[3]: bias for tanh 2
    # params[4]: output weight for tanh 1
    # params[5]: output weight for tanh 2
    # params[6]: coeff for x^3
    # params[7]: coeff for x^2
    # params[8]: coeff for x
    # params[9]: constant term
    
    h1 = np.tanh(params[0] * x + params[1])
    h2 = np.tanh(params[2] * x + params[3])
    
    poly = (params[6] * x**3 + 
            params[7] * x**2 + 
            params[8] * x + 
            params[9])
            
    return params[4] * h1 + params[5] * h2 + poly
