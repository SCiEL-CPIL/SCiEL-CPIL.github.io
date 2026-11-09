import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a rational (Pade-like) form:
    y = (p0 + p1*x + p2*v + p3*x^2 + p4*v^2) / (1 + p5*x + p6*v + p7*x^2 + p8*v^2)
    with p9 as an additive constant.
    """
    # Scale inputs for better conditioning
    x_s = x / 0.55
    v_s = v / 0.23
    
    # Numerator terms (5 params)
    num = (params[0] + 
           params[1] * x_s + 
           params[2] * v_s + 
           params[3] * x_s**2 + 
           params[4] * v_s**2)
    
    # Denominator terms (4 params, +1 offset)
    den = (1.0 + 
           params[5] * x_s + 
           params[6] * v_s + 
           params[7] * x_s**2 + 
           params[8] * v_s**2)
    
    # Prevent division by zero
    den_safe = np.where(np.abs(den) < 1e-8, 1e-8, den)
    
    # Additive constant (1 param)
    result = num / den_safe + params[9]
    
    return result
