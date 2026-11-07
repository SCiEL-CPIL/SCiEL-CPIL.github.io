import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize inputs to roughly [0, 1]
    # b: [1, 4.998] -> [0, 1]
    b_n = (b - 1.0) / 4.0
    # s: [1, 4.999] -> [0, 1]
    s_n = (s - 1.0) / 4.0
    # temp: [1, 39.98] -> [0, 1]
    t_n = (temp - 1.0) / 39.0
    # pH: [2, 12] -> [0, 1]
    ph_n = (pH - 2.0) / 10.0

    p = params

    # Unpack params
    mu_max = p[0]
    
    # Centers for Gaussian factors, centered around 0.5
    c_T = 0.5 + np.tanh(p[1])
    c_pH = 0.5 + np.tanh(p[3])
    
    # Widths for Gaussian factors, scaled to be small initially
    # exp(1) ~ 2.7, /5 ~ 0.54. Reasonable width for [0,1] range.
    w_T = np.exp(p[2]) / 5.0
    w_pH = np.exp(p[4]) / 5.0
    
    # Monod Ks
    K_s = np.exp(p[5]) / 5.0
    
    # Biomass inhibition K
    K_b = np.exp(p[6]) / 5.0
    
    # Offset
    offset = p[7]
    
    # Additive interaction coefficients
    a1 = p[8]
    a2 = p[9]

    # Calculate factors
    # Gaussian temp factor
    temp_factor = np.exp(-((t_n - c_T) ** 2) / (2.0 * w_T ** 2))
    
    # Gaussian pH factor
    ph_factor = np.exp(-((ph_n - c_pH) ** 2) / (2.0 * w_pH ** 2))
    
    # Monod substrate factor
    sub_factor = s_n / (K_s + s_n)
    
    # Biomass inhibition factor
    bio_factor = 1.0 / (1.0 + b_n / K_b)

    # Combine multiplicatively
    base_growth = mu_max * temp_factor * ph_factor * sub_factor * bio_factor
    
    # Add additive terms
    additive = a1 * b_n * s_n + a2 * t_n * ph_n
    
    return base_growth + additive + offset
