import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model.
    
    Parameters:
    b: population density
    s: substrate concentration
    temp: temperature
    pH: pH level
    params: array of 10 parameters
        p[0]: mu_max
        p[1]: K_s (Monod half-saturation)
        p[2]: T_opt (Optimal Temperature)
        p[3]: sigma_T (Temperature width)
        p[4]: pH_opt (Optimal pH)
        p[5]: sigma_pH (pH width)
        p[6]: K_b (Density inhibition constant)
        p[7]: Monod exponent
        p[8]: Inhibition exponent
        p[9]: Offset
    """
    p = params
    
    # Extract parameters
    mu_max = p[0]
    K_s = p[1]
    T_opt = p[2]
    sigma_T = p[3]
    pH_opt = p[4]
    sigma_pH = p[5]
    K_b = p[6]
    monod_exp = p[7]
    inh_exp = p[8]
    offset = p[9]
    
    # Ensure positive values for denominators and variances to avoid NaNs
    # Using a small epsilon
    eps = 1e-6
    
    K_s_safe = np.maximum(K_s, eps)
    sigma_T_safe = np.maximum(np.abs(sigma_T), eps)
    sigma_pH_safe = np.maximum(np.abs(sigma_pH), eps)
    K_b_safe = np.maximum(K_b, eps)
    
    # Monod term for substrate
    # s is in [1, 40], K_s likely around 5-10
    monod_term = (s / (K_s_safe + s)) ** monod_exp
    
    # Temperature term (Gaussian)
    # temp in [1, 40], T_opt ~ 37
    temp_term = np.exp(-((temp - T_opt) ** 2) / (2 * sigma_T_safe ** 2))
    
    # pH term (Gaussian)
    # pH in [2, 12], pH_opt ~ 7
    ph_term = np.exp(-((pH - pH_opt) ** 2) / (2 * sigma_pH_safe ** 2))
    
    # Density inhibition term
    # b in [1, 5], inhibition should reduce growth
    # Using 1 / (1 + (b / K_b)^inh_exp)
    # Or simpler: 1 / (1 + K_b * b)
    # Let's use a flexible power law inhibition: (1 / (1 + (b / K_b_safe))) ** inh_exp
    # If K_b is small, inhibition is strong.
    # b is positive.
    b_term = (1.0 / (1.0 + b / K_b_safe)) ** inh_exp
    
    # Combine
    mu = mu_max * monod_term * temp_term * ph_term * b_term + offset
    
    return mu
