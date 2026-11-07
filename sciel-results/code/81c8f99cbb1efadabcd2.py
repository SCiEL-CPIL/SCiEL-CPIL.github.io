import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize inputs to [0, 1] based on training ranges
    # b: [1, 5] -> (b-1)/4
    # s: [1, 5] -> (s-1)/4
    # temp: [1, 40] -> (temp-1)/39
    # pH: [2, 12] -> (pH-2)/10
    
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    t_norm = (temp - 1.0) / 39.0
    ph_norm = (pH - 2.0) / 10.0
    
    p = params
    
    # Structural Change: Multiplicative model (product of factors)
    # Growth rate = C * f_substrate * f_temp * f_ph * f_biomass
    # This better captures the interaction where growth is 0 if any factor is 0.
    
    # 1. Substrate: Monod-like saturation
    # p[0]: Ks (half-saturation) - squared for positivity
    # p[1]: Max rate scale for substrate (effectively absorbed into C)
    k_s = p[0]**2 + 1e-6
    term_s = s_norm / (s_norm + k_s)
    
    # 2. Temperature: Gaussian bell
    # p[2]: Center (normalized)
    # p[3]: Width - squared for positivity
    t_center = p[2]
    t_width = p[3]**2 + 1e-6
    term_t = np.exp(-((t_norm - t_center)**2) / (2 * t_width))
    
    # 3. pH: Gaussian bell
    # p[4]: Center (normalized)
    # p[5]: Width - squared for positivity
    ph_center = p[4]
    ph_width = p[5]**2 + 1e-6
    term_ph = np.exp(-((ph_norm - ph_center)**2) / (2 * ph_width))
    
    # 4. Biomass: Linear
    # p[6]: Biomass Coefficient
    # p[7]: Biomass Offset (to allow non-zero intercept or shift)
    term_b = p[6] * b_norm + p[7]
    
    # 5. Global Scale
    # p[8]: Overall Amplitude
    # p[9]: Offset (Additive background)
    amplitude = p[8]
    offset = p[9]
    
    # Combine Multiplicatively for the main terms, additively for offset
    # y = offset + amplitude * term_s * term_t * term_ph * term_b
    
    y = offset + amplitude * term_s * term_t * term_ph * term_b
    
    # Ensure finite output
    if not np.all(np.isfinite(y)):
        return np.zeros_like(y)
        
    return y
