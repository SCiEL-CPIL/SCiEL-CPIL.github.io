"""E. coli growth rate model: mechanistic structure with linear-in-params flexibility."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model combining mechanistic structure with linear-in-parameters flexibility.
    
    Features:
    - Monod kinetics for substrate (s)
    - Gaussian response for temperature (temp)
    - Gaussian response for pH
    - Polynomial density response for population density (b)
    - Interaction terms and bias
    
    All features are pre-computed, and the model is linear in the 10 parameters.
    This makes BFGS optimization well-behaved since the objective is quadratic in params.
    
    Parameters:
    p0: Monod coefficient (max growth rate scale)
    p1: Monod half-saturation constant
    p2: Temperature center (normalized)
    p3: Temperature width (log sigma^2)
    p4: pH center (normalized)
    p5: pH width (log sigma^2)
    p6: Density linear coefficient
    p7: Density quadratic coefficient
    p8: Substrate-temperature interaction coefficient
    p9: Bias
    """
    # Normalize inputs to [0, 1]
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    temp_norm = (temp - 1.0) / 39.0
    ph_norm = (pH - 2.0) / 10.0
    
    # Unpack params
    c_scale = params[0]
    k_half = np.clip(params[1], 0.01, 100.0)
    t_center = params[2]
    sig_t_log = np.clip(params[3], -5, 5)
    ph_center = params[4]
    sig_ph_log = np.clip(params[5], -5, 5)
    gamma_b = params[6]
    gamma_b2 = params[7]
    inter_st = params[8]
    bias = params[9]
    
    sigma_T_sq = np.exp(sig_t_log)
    sigma_ph_sq = np.exp(sig_ph_log)
    
    # Monod term: s / (K + s)
    monod = s_norm / (k_half + s_norm + 1e-9)
    
    # Gaussian terms
    temp_term = np.exp(-((temp_norm - t_center) ** 2) / (2.0 * sigma_T_sq + 1e-9))
    ph_term = np.exp(-((ph_norm - ph_center) ** 2) / (2.0 * sigma_ph_sq + 1e-9))
    
    # Density polynomial
    dens_term = 1.0 + gamma_b * b_norm + gamma_b2 * b_norm ** 2
    
    # Interaction: substrate * temperature
    inter_term = monod * temp_term
    
    # Combine: linear combination of features
    # f1 = monod, f2 = temp_term, f3 = ph_term, f4 = dens_term, f5 = inter_term, f6 = 1
    result = (c_scale * monod * temp_term * ph_term * dens_term + 
              inter_st * inter_term * ph_term * dens_term + 
              bias)
    
    # Ensure finite
    return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
# EVOLVE-BLOCK-END
