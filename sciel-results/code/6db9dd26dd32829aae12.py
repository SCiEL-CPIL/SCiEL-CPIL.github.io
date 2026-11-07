"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using:
    - Monod kinetics for substrate
    - Gaussian response for temperature
    - Gaussian response for pH
    - Inhibition for population density
    
    Parameters (10 total):
    p0: max growth rate
    p1: Monod half-saturation constant (Ks)
    p2: optimal temperature (T_opt)
    p3: temperature sensitivity (sigma_T)
    p4: optimal pH (pH_opt)
    p5: pH sensitivity (sigma_pH)
    p6: inhibition constant for density (Ki)
    p7: inhibition exponent
    p8: baseline growth offset
    p9: additional scaling factor
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure positive constants where needed
    Ks = np.abs(p1) + 0.01
    sigma_T = np.abs(p3) + 0.1
    sigma_pH = np.abs(p5) + 0.1
    Ki = np.abs(p6) + 0.01
    
    # Monod kinetics for substrate
    monod_s = s / (Ks + s)
    
    # Gaussian response for temperature
    temp_response = np.exp(-((temp - p2) ** 2) / (2 * sigma_T ** 2))
    
    # Gaussian response for pH
    pH_response = np.exp(-((pH - p4) ** 2) / (2 * sigma_pH ** 2))
    
    # Inhibition for population density
    density_inhibition = 1.0 / (1.0 + (b / Ki) ** p7)
    
    # Combine all terms
    growth = p0 * p9 * monod_s * temp_response * pH_response * density_inhibition + p8
    
    return growth
# EVOLVE-BLOCK-END
