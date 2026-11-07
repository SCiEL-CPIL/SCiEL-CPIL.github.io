"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using Monod kinetics with temperature and pH dependencies.
    
    Structure:
    mu = mu_max * (S / (Ks + S)) * f_temp * f_ph
    
    where:
    - mu_max = params[0]
    - Ks = params[1] (half-saturation constant for substrate)
    - f_temp = exp(-params[2] * (temp - T_opt)^2)  (Gaussian temperature dependence)
    - T_opt = params[3] (optimal temperature)
    - f_ph = exp(-params[4] * (pH - pH_opt)^2)    (Gaussian pH dependence)
    - pH_opt = params[5] (optimal pH)
    - params[6] = additional scaling for substrate effect
    - params[7] = inhibition term coefficient
    - params[8] = biomass-dependent inhibition (density-dependent)
    - params[9] = baseline offset
    
    Using exp for positive parameters where needed to avoid negative values during optimization.
    """
    p = params
    
    # Extract parameters
    mu_max = p[0]
    Ks = np.exp(p[1])  # Ensure Ks > 0
    temp_width = np.exp(p[2])  # Ensure width > 0
    T_opt = p[3]
    ph_width = np.exp(p[4])  # Ensure width > 0
    pH_opt = p[5]
    sub_scale = p[6]
    inhib_coeff = p[7]
    density_inhib = p[8]
    offset = p[9]
    
    # Monod term for substrate
    # S is substrate concentration
    monod = s / (Ks + s)
    
    # Temperature dependence (Gaussian)
    temp_effect = np.exp(-((temp - T_opt) ** 2) / (2.0 * temp_width ** 2 + 1e-8))
    
    # pH dependence (Gaussian)
    ph_effect = np.exp(-((pH - pH_opt) ** 2) / (2.0 * ph_width ** 2 + 1e-8))
    
    # Density-dependent inhibition (quasi-logistic)
    # b is population density
    density_term = 1.0 / (1.0 + density_inhib * b)
    
    # Substrate inhibition at high concentrations
    sub_inhib = 1.0 / (1.0 + inhib_coeff * s)
    
    # Combine all effects
    growth = mu_max * monod * temp_effect * ph_effect * density_term * sub_inhib * sub_scale
    
    # Add baseline
    growth = growth + offset
    
    return growth
# EVOLVE-BLOCK-END
