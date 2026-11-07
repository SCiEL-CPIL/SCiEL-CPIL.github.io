import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize inputs to [0, 1]
    # b: [1, 5] -> (b-1)/4
    # s: [1, 5] -> (s-1)/4
    # temp: [1, 40] -> (temp-1)/39
    # pH: [2, 12] -> (pH-2)/10
    
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    t_norm = (temp - 1.0) / 39.0
    ph_norm = (pH - 2.0) / 10.0
    
    p = params
    
    # Ensure positive parameters for widths and constants
    # Using a small epsilon to avoid division by zero or negative widths
    eps = 1e-6
    
    # p[1]: Substrate saturation constant (Ks)
    p1 = np.maximum(p[1], eps)
    
    # p[2]: Temperature width (inverse variance)
    p2 = np.maximum(p[2], eps)
    # p[3]: Temperature optimum
    p3 = p[3]
    
    # p[4]: pH width
    p4 = np.maximum(p[4], eps)
    # p[5]: pH optimum
    p5 = p[5]
    
    # p[6]: Biomass inhibition strength
    p6 = np.maximum(p[6], eps)
    
    # Calculate terms
    # Monod-like substrate term
    term_s = s_norm / (s_norm + p1)
    
    # Gaussian temperature term
    term_t = np.exp(-p2 * (t_norm - p3)**2)
    
    # Gaussian pH term
    term_ph = np.exp(-p4 * (ph_norm - p5)**2)
    
    # Biomass inhibition term (exponential decay)
    term_b = np.exp(-p6 * b_norm**2)
    
    # Linear corrections to capture non-multiplicative effects
    # p[7]: Bias
    # p[8]: Linear temp correction
    # p[9]: Linear pH correction
    linear_corr = p[7] + p[8] * t_norm + p[9] * ph_norm
    
    # Combine
    # Main multiplicative term
    main_term = p[0] * term_s * term_t * term_ph * term_b
    
    # Final prediction
    y = main_term + linear_corr
    
    return y
