"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Autonomous nonlinear oscillator model with normalized inputs.
    
    Physics prior:
    - No explicit time dependence (autonomous system).
    - Potential force: odd powers of x (cubic/fifth-order oscillator).
    - Damping: odd powers of v (linear, cubic).
    - Cross terms for parametric damping/coupling.
    
    Normalization: x and v are scaled to unit variance to improve
    the condition number of the Hessian for BFGS convergence.
    
    Basis functions (10 params):
    0: 1
    1: x_n
    2: x_n^3
    3: x_n^5
    4: v_n
    5: v_n^3
    6: x_n * v_n
    7: x_n^2 * v_n
    8: x_n * v_n^2
    9: v_n^2
    """
    p = params
    
    # Normalize inputs to improve conditioning
    # Using typical training range centers and scales
    x_center = 0.0
    x_scale = 0.35  # approx half-range of x
    v_center = 0.0
    v_scale = 0.35  # approx half-range of v
    
    x_n = (x - x_center) / x_scale
    v_n = (v - v_center) / v_scale
    
    # Precompute powers
    x2 = x_n * x_n
    x3 = x2 * x_n
    x5 = x3 * x2
    
    v2 = v_n * v_n
    v3 = v2 * v_n
    
    # Terms
    term0 = p[0]
    term1 = p[1] * x_n
    term2 = p[2] * x3
    term3 = p[3] * x5
    term4 = p[4] * v_n
    term5 = p[5] * v3
    term6 = p[6] * x_n * v_n
    term7 = p[7] * x2 * v_n
    term8 = p[8] * x_n * v2
    term9 = p[9] * v2
    
    return term0 + term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9
# EVOLVE-BLOCK-END
