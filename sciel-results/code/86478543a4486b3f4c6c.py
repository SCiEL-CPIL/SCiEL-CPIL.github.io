"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """E. coli growth rate from population density, substrate, temperature and pH."""
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Monod kinetics for substrate: s / (Ks + s)
    Ks = p1**2 + 1e-6
    monod_s = s / (Ks + s)
    
    # Gaussian temperature optimum
    T_opt = p2
    T_width = p3**2 + 1e-6
    temp_factor = np.exp(-((temp - T_opt)**2) / (2.0 * T_width))
    
    # Gaussian pH optimum
    pH_opt = p4
    pH_width = p5**2 + 1e-6
    ph_factor = np.exp(-((pH - pH_opt)**2) / (2.0 * pH_width))
    
    # Density effect: quadratic in b
    dens_term = 1.0 + p6 * b + p7 * b**2
    
    # Combine: max growth * Monod * temp * pH * density
    max_growth = p0
    result = max_growth * monod_s * temp_factor * ph_factor * dens_term + p8 * b + p9
    
    return result
# EVOLVE-BLOCK-END
