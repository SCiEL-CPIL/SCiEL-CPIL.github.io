import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Extract params
    # 1. mu_max (positive)
    # 2. K_s (positive)
    # 3. T_opt
    # 4. sigma_T (positive)
    # 5. pH_opt
    # 6. sigma_pH (positive)
    # 7. b_coeff
    # 8. b_intercept
    # 9. T_pH_interact
    # 10. bias
    
    mu_max = np.exp(params[0])
    K_s = np.exp(params[1])
    T_opt = params[2]
    sigma_T = np.exp(params[3])
    pH_opt = params[4]
    sigma_pH = np.exp(params[5])
    b_coeff = params[6]
    b_intercept = params[7]
    T_pH_interact = params[8]
    bias = params[9]
    
    # Monod term for substrate
    # s is in [1, 5]. K_s should be comparable.
    monod_s = s / (K_s + s)
    
    # Gaussian term for temperature
    # temp is in [1, 40].
    # Use normalized temp for stability in exponent?
    # (temp - T_opt) / sigma_T
    # If sigma_T is small, this explodes. 
    # Clip the exponent argument to prevent overflow?
    # Or use a softer function.
    # Let's use np.exp(-0.5 * ((temp - T_opt) / sigma_T) ** 2)
    # To be safe, clip the squared term.
    z_T = (temp - T_opt) / sigma_T
    z_T = np.clip(z_T, -10, 10)
    gauss_T = np.exp(-0.5 * z_T ** 2)
    
    # Gaussian term for pH
    z_pH = (pH - pH_opt) / sigma_pH
    z_pH = np.clip(z_pH, -10, 10)
    gauss_pH = np.exp(-0.5 * z_pH ** 2)
    
    # Linear term for biomass
    # b is in [1, 5]. Center at 3.
    b_norm = (b - 3.0) / 2.0
    b_term = b_intercept + b_coeff * b_norm
    
    # Interaction term T * pH (normalized)
    T_norm = (temp - 20.0) / 10.0
    pH_norm = (pH - 7.0) / 5.0
    inter = T_pH_interact * (T_norm * pH_norm)
    
    mu = mu_max * monod_s * gauss_T * gauss_pH * b_term + inter + bias
    
    return mu
