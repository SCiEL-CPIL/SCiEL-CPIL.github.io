"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Approach: Product of two affine functions + symmetric interaction + constant
    # f(x,y,z) = (p0 + p1*x + p2*y + p3*z) * (p4 + p5*x + p6*y + p7*z) 
    #           + p8*(x*y + y*z + x*z) + p9
    # This is a structured quadratic model, distinct from rational or pure polynomial basis.
    
    p = params
    
    # First affine combination
    a = p[0] + p[1] * x + p[2] * y + p[3] * z
    
    # Second affine combination
    b = p[4] + p[5] * x + p[6] * y + p[7] * z
    
    # Symmetric pairwise interaction term
    interaction = x * y + y * z + x * z
    
    # Final prediction
    result = a * b + p[8] * interaction + p[9]
    
    return result
# EVOLVE-BLOCK-END
