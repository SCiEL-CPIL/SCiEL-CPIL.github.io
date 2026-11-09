"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """E. coli growth rate from population density, substrate, temperature and pH."""
    # Normalize inputs to [0, 1] range for stable optimization
    # b: [1, 4.998] -> [0, 1]
    b_norm = (b - 1.0) / 4.0
    # s: [1, 4.999] -> [0, 1]
    s_norm = (s - 1.0) / 4.0
    # temp: [1.002, 39.98] -> [0, 1]
    temp_norm = (temp - 1.0) / 39.0
    # pH: [2, 12] -> [0, 1]
    pH_norm = (pH - 2.0) / 10.0

    # Clip to [0, 1] to handle slight out-of-range values gracefully
    b_norm = np.clip(b_norm, 0.0, 1.0)
    s_norm = np.clip(s_norm, 0.0, 1.0)
    temp_norm = np.clip(temp_norm, 0.0, 1.0)
    pH_norm = np.clip(pH_norm, 0.0, 1.0)

    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Basis functions: linear + quadratic for each variable
    # b term
    b_term = p0 * b_norm + p1 * b_norm**2
    # s term
    s_term = p2 * s_norm + p3 * s_norm**2
    # temp term
    temp_term = p4 * temp_norm + p5 * temp_norm**2
    # pH term
    ph_term = p6 * pH_norm + p7 * pH_norm**2
    # Bias
    bias = p8
    # Mild interaction: b * s (captures some multivariate structure)
    interaction = p9 * b_norm * s_norm

    return b_term + s_term + temp_term + ph_term + bias + interaction
# EVOLVE-BLOCK-END
