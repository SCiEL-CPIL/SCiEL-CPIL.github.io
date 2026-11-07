import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """E. coli growth rate from population density, substrate, temperature and pH."""
    # Unpack 10 parameters
    c0 = params[0]          # Max growth rate scale
    Ks = params[1]          # Monod half-saturation constant
    T_opt = params[2]       # Optimal temperature
    sigma_T = params[3]     # Temperature sensitivity width
    pH_opt = params[4]      # Optimal pH
    sigma_pH = params[5]    # pH sensitivity width
    alpha_b = params[6]     # Density-dependent growth modulation
    beta_b = params[7]      # Density inhibition strength
    gamma = params[8]       # Cross-term coefficient
    delta = params[9]       # Offset/baseline
    
    # Ensure positive denominators and widths for numerical stability
    Ks_safe = np.maximum(np.abs(Ks), 1e-4)
    sigma_T_safe = np.maximum(np.abs(sigma_T), 0.5)
    sigma_pH_safe = np.maximum(np.abs(sigma_pH), 0.5)
    beta_b_safe = np.maximum(np.abs(beta_b), 1e-4)
    
    # Monod substrate saturation: s / (Ks + s)
    # This gives 0 when s=0 and approaches 1 when s >> Ks
    monod = s / (Ks_safe + s)
    
    # Temperature Gaussian response: peaks at T_opt
    temp_factor = np.exp(-((temp - T_opt) ** 2) / (2.0 * sigma_T_safe ** 2))
    
    # pH Gaussian response: peaks at pH_opt
    ph_factor = np.exp(-((pH - pH_opt) ** 2) / (2.0 * sigma_pH_safe ** 2))
    
    # Density-dependent effects
    # At low density, growth is near maximal; at high density, inhibition
    # Use a smooth saturation: b / (b + beta_b) gives 0 at b=0, approaches 1
    # But we want growth to DECREASE with density, so use 1/(1 + alpha_b * b)
    density_factor = 1.0 / (1.0 + alpha_b * b)
    
    # Combine: multiplicative model
    # Growth = c0 * monod * temp_factor * ph_factor * density_factor + gamma * monod * temp_factor + delta
    growth = c0 * monod * temp_factor * ph_factor * density_factor
    growth += gamma * monod * temp_factor
    growth += delta
    
    return growth
