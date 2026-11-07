import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize inputs to [0, 1] roughly based on training ranges
    b_norm = b / 5.0
    s_norm = s / 5.0
    t_norm = temp / 40.0
    ph_norm = pH / 12.0
    
    p = params
    
    # 1. Substrate: Monod-like saturation (smooth, positive)
    # p[0] is the half-saturation constant. Ensure it stays positive for stability.
    # Using p[0]**2 + 1e-6 ensures it's always positive and smooth.
    k_s = p[0]**2 + 1e-6
    term_s = s_norm / (s_norm + k_s)
    
    # 2. Temperature: Gaussian bell curve
    # p[1] = center, p[2] = width
    # Use p[2]**2 + 1e-6 for width to ensure positivity and smoothness.
    t_center = p[1]
    t_width = p[2]**2 + 1e-6
    term_t = np.exp(-((t_norm - t_center)**2) / (2 * t_width))
    
    # 3. pH: Gaussian bell curve
    # p[3] = center, p[4] = width
    ph_center = p[3]
    ph_width = p[4]**2 + 1e-6
    term_ph = np.exp(-((ph_norm - ph_center)**2) / (2 * ph_width))
    
    # 4. Biomass: Quadratic term
    # p[5], p[6], p[7] are coefficients for 1, b, b^2
    term_b = p[5] + p[6] * b_norm + p[7] * (b_norm**2)
    
    # 5. Combined model
    # Multiplicative for s, t, ph (growth requires suitable conditions)
    # But allow an additive offset p[8] and a scaling factor p[9]
    # This structure: y = p[9] * (term_s * term_t * term_ph) * term_b + p[8]
    # This is similar to the previous one but with Gaussian bells instead of Lorentzian,
    # which are more biologically plausible for temperature/pH optima and decay.
    
    y = p[9] * term_s * term_t * term_ph * term_b + p[8]
    
    return y
