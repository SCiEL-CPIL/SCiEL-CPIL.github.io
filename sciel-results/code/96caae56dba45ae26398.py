# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression candidate: A Fourier (Trigonometric) basis in x and y.
    
    This approach uses a fixed set of sine and cosine terms up to frequency 2.
    Unlike polynomials, this basis is well-suited for smooth functions on bounded
    domains and can capture periodic or oscillatory behaviors that polynomials might
    struggle with or require higher degrees to approximate.
    
    Basis functions (10 total):
      1. 1
      2. sin(pi * x)
      3. cos(pi * x)
      4. sin(pi * y)
      5. cos(pi * y)
      6. sin(pi * x) * cos(pi * y)
      7. cos(pi * x) * sin(pi * y)
      8. sin(2 * pi * x)
      9. cos(2 * pi * x)
      10. sin(2 * pi * y)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute trigonometric terms
    pi_x = np.pi * x
    pi_y = np.pi * y
    two_pi_x = 2.0 * np.pi * x
    two_pi_y = 2.0 * np.pi * y
    
    sin_px = np.sin(pi_x)
    cos_px = np.cos(pi_x)
    sin_py = np.sin(pi_y)
    cos_py = np.cos(pi_y)
    
    sin_2px = np.sin(two_pi_x)
    cos_2px = np.cos(two_pi_x)
    sin_2py = np.sin(two_pi_y)
    
    # Interaction terms
    sin_px_cos_py = sin_px * cos_py
    cos_px_sin_py = cos_px * sin_py
    
    # Linear combination
    pred = (p0 
            + p1 * sin_px 
            + p2 * cos_px 
            + p3 * sin_py 
            + p4 * cos_py 
            + p5 * sin_px_cos_py 
            + p6 * cos_px_sin_py 
            + p7 * sin_2px 
            + p8 * cos_2px 
            + p9 * sin_2py)
            
    return pred
# EVOLVE-BLOCK-END
