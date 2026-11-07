"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model using a basis of bounded nonlinearities.
    
    Uses tanh and sin for bounded, smooth nonlinearities which are well-conditioned
    for BFGS optimization compared to high-order polynomials.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Bounded nonlinearities
    tanh_x = np.tanh(x)
    tanh_v = np.tanh(v)
    sin_x = np.sin(x)
    sin_v = np.sin(v)
    
    # Interactions
    xv = x * v
    x_tanh_v = x * tanh_v
    v_tanh_x = v * tanh_x
    
    # Construct the linear combination of basis functions
    # p0: x
    # p1: v
    # p2: tanh(x)
    # p3: tanh(v)
    # p4: sin(x)
    # p5: sin(v)
    # p6: x*v
    # p7: x*tanh(v)
    # p8: v*tanh(x)
    # p9: bias
    
    return (p0 * x + 
            p1 * v + 
            p2 * tanh_x + 
            p3 * tanh_v + 
            p4 * sin_x + 
            p5 * sin_v + 
            p6 * xv + 
            p7 * x_tanh_v + 
            p8 * v_tanh_x + 
            p9)
# EVOLVE-BLOCK-END
