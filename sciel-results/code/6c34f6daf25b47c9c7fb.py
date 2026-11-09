"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """E. coli growth rate from population density, substrate, temperature and pH."""
    # Unpack 10 parameters
    # p0: intercept
    # p1: linear coeff for b
    # p2: hinge coeff for b at 3.0
    # p3: linear coeff for s
    # p4: hinge coeff for s at 5.0
    # p5: linear coeff for temp
    # p6: hinge coeff for temp at 25.0
    # p7: hinge coeff for temp at 35.0
    # p8: linear coeff for pH
    # p9: hinge coeff for pH at 7.0
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Hinge functions: max(0, x - c)
    b_hinge = np.maximum(0.0, b - 3.0)
    s_hinge = np.maximum(0.0, s - 5.0)
    temp_hinge1 = np.maximum(0.0, temp - 25.0)
    temp_hinge2 = np.maximum(0.0, temp - 35.0)
    ph_hinge = np.maximum(0.0, pH - 7.0)
    
    # Piecewise-linear model
    result = (p0 
              + p1 * b + p2 * b_hinge
              + p3 * s + p4 * s_hinge
              + p5 * temp + p6 * temp_hinge1 + p7 * temp_hinge2
              + p8 * pH + p9 * ph_hinge)
    
    return result
# EVOLVE-BLOCK-END
