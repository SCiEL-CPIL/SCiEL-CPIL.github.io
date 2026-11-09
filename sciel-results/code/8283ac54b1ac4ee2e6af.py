"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # 5s safety margin from 150s limit


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate using a 2nd-order polynomial response surface in scaled inputs.
    
    This structural change replaces the mechanistic multiplicative model with a 
    data-driven empirical polynomial. It uses 10 parameters to fit a subset of 
    quadratic terms and one interaction term, providing a flexible, smooth 
    surface without the "dead zones" of Gaussian bells.
    
    Inputs are scaled to [0, 1] based on known training ranges:
    b: [1, 5]
    s: [1, 5]
    temp: [1, 40]
    pH: [2, 12]
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        return np.zeros_like(b)

    # Scale inputs to [0, 1] to improve conditioning for the polynomial fit
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    t_norm = (temp - 1.0) / 39.0
    p_norm = (pH - 2.0) / 10.0
    
    # Clip to [0, 1] to prevent extrapolation issues if test data is slightly out of range
    b_norm = np.clip(b_norm, 0.0, 1.0)
    s_norm = np.clip(s_norm, 0.0, 1.0)
    t_norm = np.clip(t_norm, 0.0, 1.0)
    p_norm = np.clip(p_norm, 0.0, 1.0)
    
    # Construct basis functions (10 terms)
    # 1. Constant
    # 2-5. Linear terms
    # 6-9. Quadratic terms
    # 10. Interaction term (temp * pH)
    
    basis = np.column_stack((
        np.ones_like(b_norm),
        b_norm,
        s_norm,
        t_norm,
        p_norm,
        b_norm ** 2,
        s_norm ** 2,
        t_norm ** 2,
        p_norm ** 2,
        t_norm * p_norm
    ))
    
    # Linear combination of basis functions
    mu = basis @ params
    
    return mu
# EVOLVE-BLOCK-END
