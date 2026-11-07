"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Richer symbolic regression basis with 10 parameters:
    - Polynomial up to degree 4 (5 terms)
    - Sin with learnable frequency and amplitude (2 params)
    - Cos with learnable frequency and amplitude (2 params)
    - Exp with learnable rate (1 param)
    
    params layout (10 elements):
      params[0]:  a0 (constant)
      params[1]:  a1 (x coeff)
      params[2]:  a2 (x^2 coeff)
      params[3]:  a3 (x^3 coeff)
      params[4]:  a4 (x^4 coeff)
      params[5]:  b1 (sin amplitude)
      params[6]:  f1 (sin frequency)
      params[7]:  b2 (cos amplitude)
      params[8]:  f2 (cos frequency)
      params[9]:  c1 (exp rate)
    """
    a0 = params[0]
    a1 = params[1]
    a2 = params[2]
    a3 = params[3]
    a4 = params[4]
    b1 = params[5]
    f1 = params[6]
    b2 = params[7]
    f2 = params[8]
    c1 = params[9]
    
    # Polynomial part (degree 4)
    poly = a0 + a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4
    
    # Trigonometric part - clip frequencies to avoid numerical issues
    f1_clipped = np.clip(f1, -15, 15)
    f2_clipped = np.clip(f2, -15, 15)
    trig = b1 * np.sin(f1_clipped * x) + b2 * np.cos(f2_clipped * x)
    
    # Exponential part - clip rate to avoid overflow
    c1_clipped = np.clip(c1, -2, 2)
    exp_term = np.exp(c1_clipped * x)
    
    result = poly + trig + exp_term
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
