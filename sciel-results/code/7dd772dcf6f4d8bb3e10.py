# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression candidate: A Fourier series basis in x and y.
    This approach models the target function as a sum of sines and cosines.
    Unlike polynomials, Fourier bases are global and periodic, which can
    better capture oscillatory behavior or smooth non-polynomial shapes
    on the bounded domain [-1, 1].
    
    Basis functions (10 total):
    1. 1
    2. cos(pi * x)
    3. sin(pi * x)
    4. cos(pi * y)
    5. sin(pi * y)
    6. cos(pi * x) * cos(pi * y)
    7. sin(pi * x) * sin(pi * y)
    8. cos(2 * pi * x)
    9. sin(2 * pi * x)
    10. cos(2 * pi * y)
    
    The model is linear in the parameters, ensuring stable convergence with BFGS.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute trigonometric terms
    cos_pi_x = np.cos(np.pi * x)
    sin_pi_x = np.sin(np.pi * x)
    cos_pi_y = np.cos(np.pi * y)
    sin_pi_y = np.sin(np.pi * y)
    
    # Second harmonic
    cos_2pi_x = np.cos(2 * np.pi * x)
    sin_2pi_x = np.sin(2 * np.pi * x)
    cos_2pi_y = np.cos(2 * np.pi * y)
    
    # Interaction terms
    cos_x_cos_y = cos_pi_x * cos_pi_y
    sin_x_sin_y = sin_pi_x * sin_pi_y
    
    # Linear combination
    pred = (p0 
            + p1 * cos_pi_x 
            + p2 * sin_pi_x 
            + p3 * cos_pi_y 
            + p4 * sin_pi_y 
            + p5 * cos_x_cos_y 
            + p6 * sin_x_sin_y 
            + p7 * cos_2pi_x 
            + p8 * sin_2pi_x 
            + p9 * cos_2pi_y)
            
    return pred
# EVOLVE-BLOCK-END
