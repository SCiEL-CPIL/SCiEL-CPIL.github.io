"""E. coli growth rate: mechanistic Monod + Gaussian env + density inhibition."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate from population density, substrate, temperature and pH.
    
    Model:
        mu = mu_max * (S/(Ks+S)) * exp(-((T-T_opt)^2)/(2*sigma_T^2)) 
             * exp(-((pH-pH_opt)^2)/(2*sigma_pH^2)) * 1/(1+Kb*max(0, b-b_min))
    
    Parameters (10 total):
        p[0]: mu_max (max specific growth rate)
        p[1]: Ks (half-saturation constant for substrate)
        p[2]: T_opt (optimal temperature)
        p[3]: sigma_T (temperature tolerance width)
        p[4]: pH_opt (optimal pH)
        p[5]: sigma_pH (pH tolerance width)
        p[6]: Kb (density inhibition constant)
        p[7]: b_min (density threshold before inhibition)
        p[8]: offset (additive bias)
        p[9]: scale (multiplicative scale on whole model)
    """
    mu_max = params[0]
    Ks = params[1]
    T_opt = params[2]
    sigma_T = params[3]
    pH_opt = params[4]
    sigma_pH = params[5]
    Kb = params[6]
    b_min = params[7]
    offset = params[8]
    scale = params[9]
    
    # Ensure positive widths and Ks to avoid numerical issues
    Ks_safe = np.abs(Ks) + 1e-6
    sigma_T_safe = np.abs(sigma_T) + 1e-6
    sigma_pH_safe = np.abs(sigma_pH) + 1e-6
    Kb_safe = np.abs(Kb) + 1e-6
    
    # Monod term
    monod = s / (Ks_safe + s)
    
    # Gaussian temperature response
    temp_term = np.exp(-((temp - T_opt) ** 2) / (2.0 * sigma_T_safe ** 2))
    
    # Gaussian pH response
    ph_term = np.exp(-((pH - pH_opt) ** 2) / (2.0 * sigma_pH_safe ** 2))
    
    # Density inhibition (only above b_min)
    excess = np.maximum(0.0, b - b_min)
    inhibition = 1.0 / (1.0 + Kb_safe * excess)
    
    mu = mu_max * monod * temp_term * ph_term * inhibition
    mu = scale * mu + offset
    
    return mu
# EVOLVE-BLOCK-END
