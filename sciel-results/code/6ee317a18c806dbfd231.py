import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """E. coli growth rate from population density, substrate, temperature and pH."""
    c0, Ks, T_opt, sigma_T, pH_opt, sigma_pH, alpha_b, beta_b, gamma, delta = params
    
    # Ensure positive denominators and widths
    Ks_safe = np.maximum(np.abs(Ks), 1e-6)
    sigma_T_safe = np.maximum(np.abs(sigma_T), 0.1)
    sigma_pH_safe = np.maximum(np.abs(sigma_pH), 0.1)
    beta_b_safe = np.maximum(np.abs(beta_b), 1e-6)
    
    # Monod substrate saturation
    monod = s / (Ks_safe + s)
    
    # Temperature Gaussian
    temp_factor = np.exp(-((temp - T_opt) ** 2) / (2.0 * sigma_T_safe ** 2))
    
    # pH Gaussian
    ph_factor = np.exp(-((pH - pH_opt) ** 2) / (2.0 * sigma_pH_safe ** 2))
    
    # Density inhibition (Hill-like)
    density_factor = 1.0 / (1.0 + (b / beta_b_safe) ** alpha_b)
    
    # Combine
    growth = c0 * monod * temp_factor * ph_factor * density_factor * gamma + delta
    
    return growth
