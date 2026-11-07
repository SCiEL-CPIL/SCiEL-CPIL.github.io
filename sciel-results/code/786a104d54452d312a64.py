"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rich basis symbolic regression model with 10 learnable parameters.
    
    params layout:
      p[0]:  amplitude for x term
      p[1]:  amplitude for x^2 term
      p[2]:  amplitude for x^3 term
      p[3]:  constant term
      p[4]:  frequency for sin
      p[5]:  amplitude for sin
      p[6]:  frequency for cos
      p[7]:  amplitude for cos
      p[8]:  rate for exp
      p[9]:  amplitude for exp
    
    The model is:
      y = p0*x + p1*x^2 + p2*x^3 + p3 
        + p5*sin(p4*x) + p7*cos(p6*x) 
        + p9*exp(p8*x)
    
    To avoid numerical issues with exp for large |p8*x|, we clip.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Ensure params is 1D array of length 10
    p = np.asarray(params, dtype=np.float64).flatten()
    if p.shape[0] != 10:
        # If wrong length, pad or truncate
        if p.shape[0] < 10:
            p = np.concatenate([p, np.ones(10 - p.shape[0])])
        else:
            p = p[:10]
    
    # Polynomial terms
    y = p[0] * x + p[1] * x**2 + p[2] * x**3 + p[3]
    
    # Trigonometric terms with learnable frequencies
    # Clip frequencies to avoid extreme values that could cause issues
    freq_sin = np.clip(p[4], -10, 10)
    freq_cos = np.clip(p[6], -10, 10)
    y += p[5] * np.sin(freq_sin * x)
    y += p[7] * np.cos(freq_cos * x)
    
    # Exponential term with clipping to avoid overflow
    rate = np.clip(p[8], -5, 5)
    exp_arg = rate * x
    # Clip exp argument to avoid overflow
    exp_arg_clipped = np.clip(exp_arg, -50, 50)
    y += p[9] * np.exp(exp_arg_clipped)
    
    return y
# EVOLVE-BLOCK-END
