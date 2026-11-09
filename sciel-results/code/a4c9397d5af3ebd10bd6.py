"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator using a 3rd-order polynomial basis
    in scaled position and velocity.
    
    Inputs are scaled to [-1, 1] to improve the condition number for BFGS optimization.
    """
    # Fixed scaling constants derived from the training data range
    x_min, x_max = -0.5728, 0.5275
    v_min, v_max = -0.1754, 0.2821
    
    # Scale inputs to [-1, 1]
    xs = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
    vs = 2.0 * (v - v_min) / (v_max - v_min) - 1.0
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute polynomial terms
    # Order 0
    term0 = p0
    # Order 1
    term1 = p1 * xs + p2 * vs
    # Order 2
    term2 = p3 * xs**2 + p4 * vs**2 + p5 * xs * vs
    # Order 3
    term3 = p6 * xs**3 + p7 * vs**3 + p8 * xs**2 * vs + p9 * xs * vs**2
    
    return term0 + term1 + term2 + term3
# EVOLVE-BLOCK-END
