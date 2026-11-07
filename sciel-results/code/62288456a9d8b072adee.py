"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration model.
    Uses a polynomial basis in scaled coordinates to capture nonlinear dynamics.
    The model is designed to be well-conditioned for BFGS optimization.
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Scale inputs to roughly [-1, 1] to improve conditioning
    # Based on training ranges: x in [-0.5728, 0.5275], v in [-0.1754, 0.2821]
    # Center and scale
    x_c = x * 1.8  # Approx scaling to get range ~[-1, 1]
    v_c = v * 5.0  # Approx scaling to get range ~[-1, 1]
    
    # Precompute powers and cross terms
    x2 = x_c * x_c
    x3 = x2 * x_c
    v2 = v_c * v_c
    v3 = v2 * v_c
    xv = x_c * v_c
    
    # Model: combination of linear, quadratic, and cubic terms
    # We use all 10 parameters for a rich but not over-parameterized basis
    # Terms:
    # 1. Constant (offset)
    # 2. Linear x
    # 3. Linear v
    # 4. x^2
    # 5. v^2
    # 6. x*v
    # 7. x^3
    # 8. v^3
    # 9. x^2*v
    # 10. x*v^2
    
    return (p0 + 
            p1 * x_c + 
            p2 * v_c + 
            p3 * x2 + 
            p4 * v2 + 
            p5 * xv + 
            p6 * x3 + 
            p7 * v3 + 
            p8 * x2 * v_c + 
            p9 * x_c * v2)
# EVOLVE-BLOCK-END
