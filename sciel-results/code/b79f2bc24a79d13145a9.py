"""Symbolic regression with a flexible non-linear basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Flexible non-linear approximation using a combination of tanh and sin basis functions.
    
    Model:
    y = p0 
      + p1 * tanh(p2 * x + p3) 
      + p4 * tanh(p5 * x + p6) 
      + p7 * sin(p8 * x + p9)
      
    params layout (length 10):
      p0: constant bias
      p1, p2, p3: amplitude, scale, offset for first tanh term
      p4, p5, p6: amplitude, scale, offset for second tanh term
      p7, p8, p9: amplitude, scale, offset for sin term
      
    This model can approximate polynomials (via Taylor expansion for small scales),
    sigmoidal functions, and periodic functions, making it more versatile than
    pure polynomial or simple rational approximations.
    """
    x = np.asarray(x, dtype=float)
    
    # Extract parameters
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    # Compute basis functions
    # Adding small epsilon to scale parameters to avoid potential division by zero 
    # in gradients if the optimizer tries to go to exactly 0, though tanh/sin 
    # are smooth everywhere. The main risk is flat regions if scale is 0.
    # However, starting from 1, it should be fine.
    
    term1 = p1 * np.tanh(p2 * x + p3)
    term2 = p4 * np.tanh(p5 * x + p6)
    term3 = p7 * np.sin(p8 * x + p9)
    
    result = p0 + term1 + term2 + term3
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
