"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """E. coli growth rate from population density, substrate, temperature and pH."""
    mu_max = params[0]
    Ks = params[1]
    T_opt = params[2]
    sigma_T = np.abs(params[3]) + 1e-6
    pH_opt = params[4]
    sigma_pH = np.abs(params[5]) + 1e-6
    K_b = np.abs(params[6]) + 1e-6
    alpha_b = params[7]
    gamma = params[8]
    offset = params[9]

    # Monod substrate term
    S_term = s / (Ks + s)

    # Gaussian temperature response
    T_term = np.exp(-((temp - T_opt) ** 2) / (2.0 * sigma_T ** 2))

    # Gaussian pH response
    pH_term = np.exp(-((pH - pH_opt) ** 2) / (2.0 * sigma_pH ** 2))

    # Density term: saturating with possible inhibition
    b_term = b / (K_b + b)

    # Combine multiplicatively with a power on density term
    growth = mu_max * S_term * T_term * pH_term * (b_term ** alpha_b)

    return gamma * growth + offset
# EVOLVE-BLOCK-END
