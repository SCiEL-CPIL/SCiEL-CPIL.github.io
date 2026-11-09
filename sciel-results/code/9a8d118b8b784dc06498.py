import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model:
    y = Monod(s) + Gaussian(temp) + Gaussian(pH) + Linear(b) + Bias
    """
    # Unpack parameters
    V_max, K_s, A_t, T_opt, sigma_t, A_p, pH_opt, sigma_p, C_b, bias = params
    
    # 1. Monod term for substrate
    # Ensure K_s is positive to avoid division by zero or negative K
    # BFGS might make K_s negative. We can use abs(K_s) or K_s**2 + 1e-6
    K_s_safe = np.abs(K_s) + 1e-6
    monod = V_max * s / (K_s_safe + s)
    
    # 2. Gaussian term for temperature
    # Normalize temp to help with initial parameter values
    # temp range [1, 40], center 20, scale 10
    t_norm = (temp - 20.0) / 10.0
    # T_opt and sigma_t are in normalized units
    # Start at 1.0, which is reasonable for normalized temp
    t_diff = t_norm - T_opt
    # Ensure sigma is positive
    sigma_t_safe = np.abs(sigma_t) + 1e-6
    temp_factor = A_t * np.exp(-0.5 * (t_diff / sigma_t_safe) ** 2)
    
    # 3. Gaussian term for pH
    # pH range [2, 12], center 7, scale 5
    ph_norm = (pH - 7.0) / 5.0
    ph_diff = ph_norm - pH_opt
    sigma_p_safe = np.abs(sigma_p) + 1e-6
    ph_factor = A_p * np.exp(-0.5 * (ph_diff / sigma_p_safe) ** 2)
    
    # 4. Linear term for density
    b_term = C_b * b
    
    # Combine
    y = monod + temp_factor + ph_factor + b_term + bias
    
    return y
