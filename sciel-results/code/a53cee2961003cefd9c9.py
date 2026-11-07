import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear-in-parameters Hybrid Basis Function.
    
    This strategy relies on the fact that the model is linear in the parameters,
    making the optimization landscape convex. BFGS will reliably converge to the
    global minimum, avoiding the local minima traps of non-linear basis functions
    like RBFs with learned centers.
    
    The basis functions are chosen to cover common functional forms:
    - Polynomials (1, x, x^2, x^3) for trends and curvature.
    - Exponentials (exp(x), exp(-x)) for growth/decay.
    - Fourier terms (sin/cos at 1, 2) for periodic oscillations.
    
    Parameters (10 total):
    p[0]: coeff for 1
    p[1]: coeff for x
    p[2]: coeff for x^2
    p[3]: coeff for exp(-x)
    p[4]: coeff for exp(x)
    p[5]: coeff for sin(x)
    p[6]: coeff for cos(x)
    p[7]: coeff for sin(2x)
    p[8]: coeff for cos(2x)
    p[9]: coeff for x^3
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.ravel()
    
    p = np.asarray(params, dtype=np.float64)
    if p.ndim != 1 or p.shape[0] != 10:
        return np.zeros_like(x)
    
    # Extract parameters
    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = p
    
    # Compute basis functions
    # Polynomials
    b0 = np.ones_like(x)
    b1 = x
    b2 = x ** 2
    b9 = x ** 3
    
    # Exponentials
    # Clip x to avoid overflow in exp(x) if x gets large, though domain is [0.1, 6]
    # exp(6) ~ 403, which is safe. exp(-6) ~ 0.0025.
    b3 = np.exp(-x)
    b4 = np.exp(x)
    
    # Fourier
    b5 = np.sin(x)
    b6 = np.cos(x)
    b7 = np.sin(2 * x)
    b8 = np.cos(2 * x)
    
    # Linear combination
    result = (c0 * b0 + 
              c1 * b1 + 
              c2 * b2 + 
              c3 * b3 + 
              c4 * b4 + 
              c5 * b5 + 
              c6 * b6 + 
              c7 * b7 + 
              c8 * b8 + 
              c9 * b9)
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(result)):
        return np.zeros_like(x)
        
    return result
