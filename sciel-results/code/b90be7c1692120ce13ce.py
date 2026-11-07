import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 1 input(s).
    Uses a mixed basis of polynomials, trigonometric functions, and a saturating non-linearity.
    The 10 parameters are linear weights for these basis functions.
    """
    # Map x from [-2, 2] to [-1, 1] to normalize the domain for basis functions
    t = x / 2.0
    
    # Polynomial basis (captures trends)
    p0 = np.ones_like(t)
    p1 = t
    p2 = t**2
    p3 = t**3
    p4 = t**4
    
    # Trigonometric basis (captures periodic/oscillatory behavior)
    # Frequencies chosen to be low enough to be stable for BFGS but high enough to capture detail
    trig1 = np.sin(np.pi * t)
    trig2 = np.cos(np.pi * t)
    trig3 = np.sin(2.0 * np.pi * t)
    trig4 = np.cos(2.0 * np.pi * t)
    
    # Saturating non-linearity (captures step-like or logistic behavior)
    # tanh(2*t) provides a smooth transition around 0
    nonlin = np.tanh(2.0 * t)
    
    # Linear combination of basis functions
    y = (params[0] * p0 +
         params[1] * p1 +
         params[2] * p2 +
         params[3] * p3 +
         params[4] * p4 +
         params[5] * trig1 +
         params[6] * trig2 +
         params[7] * trig3 +
         params[8] * trig4 +
         params[9] * nonlin)
         
    return y
