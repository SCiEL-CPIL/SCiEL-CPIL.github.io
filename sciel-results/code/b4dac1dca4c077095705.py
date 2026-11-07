import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale input to [-1, 1] to improve conditioning for BFGS
    # x is in [-3, 3], so x/3 is in [-1, 1]
    u = x / 3.0
    
    # Basis functions
    # Polynomial terms
    b0 = 1.0
    b1 = u
    b2 = u**2
    b3 = u**3
    b4 = u**4
    
    # Trigonometric terms (low frequency)
    b5 = np.sin(2.0 * np.pi * u)
    b6 = np.cos(2.0 * np.pi * u)
    b7 = np.sin(4.0 * np.pi * u)
    b8 = np.cos(4.0 * np.pi * u)
    
    # Localized Gaussian term
    b9 = np.exp(-3.0 * u**2)
    
    # Combine
    pred = (params[0] * b0 + 
            params[1] * b1 + 
            params[2] * b2 + 
            params[3] * b3 + 
            params[4] * b4 + 
            params[5] * b5 + 
            params[6] * b6 + 
            params[7] * b7 + 
            params[8] * b8 + 
            params[9] * b9)
            
    return pred
