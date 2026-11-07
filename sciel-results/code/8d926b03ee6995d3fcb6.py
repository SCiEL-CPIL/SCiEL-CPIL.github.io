import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize inputs to [0, 1] based on training ranges
    # b: [1, 5] -> [0, 1]
    # s: [1, 5] -> [0, 1]
    # temp: [1, 40] -> [0, 1]
    # pH: [2, 12] -> [0, 1]
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    t_norm = (temp - 1.0) / 39.0
    ph_norm = (pH - 2.0) / 10.0
    
    p = params
    
    # 1. Substrate: Monod-like saturation
    # p[0] controls half-saturation. Squared to ensure positivity.
    k_s = p[0]**2 + 1e-6
    term_s = s_norm / (s_norm + k_s)
    
    # 2. Temperature: Gaussian bell curve
    # p[1] center, p[2] width (squared for positivity)
    t_center = p[1]
    t_width = p[2]**2 + 1e-6
    term_t = np.exp(-((t_norm - t_center)**2) / (2 * t_width))
    
    # 3. pH: Gaussian bell curve
    # p[3] center, p[4] width (squared for positivity)
    ph_center = p[3]
    ph_width = p[4]**2 + 1e-6
    term_ph = np.exp(-((ph_norm - ph_center)**2) / (2 * ph_width))
    
    # 4. Biomass: Quadratic density-dependent term
    # From Inspiration 2: Allows for non-linear response (optimum or inhibition)
    # term_b = 1.0 + p[5] * b_norm + p[6] * (b_norm - 0.5)**2
    # This is more flexible than the logistic in the parent, capturing potential
    # optima in biomass density without strict positivity constraints on coefficients.
    term_b = 1.0 + p[5] * b_norm + p[6] * (b_norm - 0.5)**2
    
    # 5. Combined Model
    # Multiplicative interaction of environmental factors and density.
    # p[7] is an additive offset.
    # p[8] is a global scaling factor.
    # p[9] introduces a cross-term interaction between temperature and pH center,
    # allowing the pH optimum to shift with temperature (Inspiration 2).
    
    # Recalculate pH term with interaction
    # ph_center_eff = p[3] + p[9] * t_norm
    # term_ph = np.exp(-((ph_norm - ph_center_eff)**2) / (2 * ph_width))
    
    # Let's stick to the simpler multiplicative structure but ensure the biomass term
    # is robust. The parent used logistic which can be stiff. Quadratic is smoother.
    
    # However, let's look at Inspiration 3 which used a Hill coefficient for substrate.
    # Parent used Monod (Hill=1). Inspiration 3 uses Hill.
    # Let's incorporate Hill coefficient for substrate using p[9]?
    # If we use p[9] for Hill, we lose the cross-term.
    # Which is more likely to help?
    # The task is E. coli growth. Substrate saturation is key.
    # Temperature and pH are usually bell-shaped.
    # Biomass density often leads to inhibition (logistic or quadratic).
    
    # Let's combine:
    # Substrate: Monod (robust, standard)
    # Temp: Gaussian
    # pH: Gaussian
    # Biomass: Quadratic (flexible)
    # Interaction: None for now, keep it simple to allow BFGS to converge easily.
    # But we have 10 params.
    # Current usage:
    # p[0]: Ks
    # p[1]: T_center
    # p[2]: T_width
    # p[3]: pH_center
    # p[4]: pH_width
    # p[5]: B_linear
    # p[6]: B_quad
    # p[7]: Offset
    # p[8]: Scale
    # p[9]: Unused?
    
    # Let's use p[9] to modulate the substrate term or as a second substrate constant?
    # Or use it for the cross-term as in Insp 2?
    # Cross-term pH-temp is biologically plausible (e.g. thermal stability of enzymes affecting pH tolerance).
    
    # Let's try the cross-term from Inspiration 2 for pH center.
    ph_center_eff = p[3] + p[9] * t_norm
    term_ph = np.exp(-((ph_norm - ph_center_eff)**2) / (2 * ph_width))
    
    y = p[8] * term_s * term_t * term_ph * term_b + p[7]
    
    return y
