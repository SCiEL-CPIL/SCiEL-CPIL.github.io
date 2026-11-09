"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """E. coli growth rate using a flexible quadratic polynomial model on normalized inputs."""
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Normalize inputs to standard range approx [-1, 1] to improve conditioning
    # Ranges from task description:
    # b: [1, 4.998] -> Center 3, Scale 2
    # s: [1, 4.999] -> Center 3, Scale 2
    # temp: [1.002, 39.98] -> Center 20.5, Scale 19.5
    # pH: [2, 12] -> Center 7, Scale 5
    
    b_norm = (b - 3.0) / 2.0
    s_norm = (s - 3.0) / 2.0
    T_norm = (temp - 20.5) / 19.5
    pH_norm = (pH - 7.0) / 5.0

    # Construct polynomial features
    # Main effects
    f1 = b_norm
    f2 = s_norm
    f3 = T_norm
    f4 = pH_norm
    
    # Squared effects (for curvature)
    f5 = T_norm ** 2
    f6 = pH_norm ** 2
    f7 = s_norm ** 2
    f8 = b_norm ** 2
    
    # Interaction term (most likely important interaction)
    f9 = T_norm * pH_norm

    # Linear combination
    y_pred = p0 + p1 * f1 + p2 * f2 + p3 * f3 + p4 * f4 + p5 * f5 + p6 * f6 + p7 * f7 + p8 * f8 + p9 * f9
    
    return y_pred
# EVOLVE-BLOCK-END
