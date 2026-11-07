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
    
    # Structural Change: Hybrid Multiplicative-Gaussian Model
    # Combines the biological multiplicative structure (zero growth if any factor is limiting)
    # with the flexibility of Gaussian bells for temperature and pH optima.
    # This avoids the rigid linear/monod assumptions while maintaining the necessary
    # zero-crossing behavior at the boundaries of the viable range.
    
    # 1. Substrate: Monod-like saturation
    # p[0]: Ks (half-saturation) - squared for positivity
    # We use a small epsilon to avoid division by zero, though s_norm >= 0.
    k_s = p[0]**2 + 1e-6
    term_s = s_norm / (s_norm + k_s)
    
    # 2. Temperature: Gaussian bell
    # p[1]: Center (normalized)
    # p[2]: Width - squared for positivity
    t_center = p[1]
    t_width = p[2]**2 + 1e-6
    term_t = np.exp(-((t_norm - t_center)**2) / (2 * t_width))
    
    # 3. pH: Gaussian bell
    # p[3]: Center (normalized)
    # p[4]: Width - squared for positivity
    ph_center = p[3]
    ph_width = p[4]**2 + 1e-6
    term_ph = np.exp(-((ph_norm - ph_center)**2) / (2 * ph_width))
    
    # 4. Biomass: Quadratic term
    # Instead of linear, allow for a potential optimum in biomass density
    # or at least a non-linear response. 
    # p[5]: Linear coefficient
    # p[6]: Quadratic coefficient
    # We shift b_norm to be centered around 0.5 to help the optimizer?
    # Actually, let's keep it simple but allow curvature.
    # term_b = 1 + p[5]*b_norm + p[6]*b_norm**2
    # To ensure it doesn't go negative easily (which would flip the sign of growth),
    # we can use a softplus or just rely on the optimizer.
    # Let's use a form that is positive: 1 + p[5]*b_norm + p[6]*(b_norm - 0.5)**2
    # This allows a peak or valley in the biomass effect.
    term_b = 1.0 + p[5] * b_norm + p[6] * (b_norm - 0.5)**2
    
    # 5. Global Scale and Offset
    # p[7]: Overall Amplitude
    # p[8]: Offset (Additive background)
    # p[9]: Interaction term? Or a second order cross term?
    # Let's use p[9] to modulate the temperature effect based on substrate availability
    # or simply as a global multiplier for the product.
    # Actually, let's use p[9] to adjust the width of the substrate term or as a second substrate constant?
    # Let's try a second substrate constant for a more complex saturation curve?
    # No, let's stick to the core structure.
    # Let's use p[9] as a "stress" factor that reduces growth at extreme biomass?
    # Or better, let's use p[9] to modulate the amplitude based on temperature?
    # Let's try: amplitude = p[7] * (1 + p[9] * t_norm)
    # This allows the max growth rate to depend on temperature linearly on top of the Gaussian shape.
    # Wait, the Gaussian already captures the temperature dependence.
    # Let's use p[9] to allow a slight shift in the pH optimum based on temperature?
    # ph_center_eff = p[3] + p[9] * t_norm
    # This is a nice interaction term.
    
    # Re-calculate term_ph with interaction
    ph_center_eff = p[3] + p[9] * t_norm
    term_ph = np.exp(-((ph_norm - ph_center_eff)**2) / (2 * ph_width))
    
    # Combine Multiplicatively for the main terms, additively for offset
    # y = offset + amplitude * term_s * term_t * term_ph * term_b
    
    amplitude = p[7]
    offset = p[8]
    
    y = offset + amplitude * term_s * term_t * term_ph * term_b
    
    # Ensure finite output
    if not np.all(np.isfinite(y)):
        return np.zeros_like(y)
        
    return y
