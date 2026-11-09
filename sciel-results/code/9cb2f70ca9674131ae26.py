"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate from population density, substrate, temperature and pH.
    Uses a smooth quadratic response surface on normalized inputs for robust BFGS convergence.
    """
    # Unpack parameters
    # 15 parameters total for full quadratic in 4 variables + intercept
    # But we only have 10 params. Let's use a reduced quadratic or specific structure.
    
    # Strategy: Use a quadratic form but constrain it to fit in 10 params.
    # y = w0 + w1*b + w2*s + w3*t + w4*p + w5*b^2 + w6*s^2 + w7*t^2 + w8*p^2 + w9*(b*s + t*p)
    # This covers main effects, pure quadratics, and one interaction term.
    
    # Alternatively, use a more biologically motivated smooth function that is smooth everywhere.
    # The previous linear model scored 1.0. The quadratic improvement scored 1.15.
    # Let's try a smooth saturation model for substrate and temperature, but ensure smoothness for BFGS.
    # Monod: s / (K + s) is smooth if K > 0.
    # Temp: Gaussian exp(-((t-Topt)/sigma)^2) is smooth.
    # pH: Gaussian exp(-((pH-pHopt)/sigma_pH)^2) is smooth.
    # Density: Maybe linear or logistic? E. coli growth is often limited by nutrients, not density directly in these simple models, 
    # but 'b' is population density. Perhaps inhibitory at high density?
    # Let's stick to the quadratic surface which was proven to improve score, but refine the parameterization.
    
    # Normalization is key for BFGS stability.
    # b range: [1, 4.998] -> center 3, scale 2 -> [-1, 1]
    # s range: [1, 39.98] -> center 20, scale 10 -> [-1.9, 2]
    # temp range: [1, 40] -> center 20, scale 10 -> [-1.9, 2]
    # pH range: [4, 10] -> center 7, scale 3 -> [-1, 1]
    
    b_norm = (b - 3.0) / 2.0
    s_norm = (s - 20.0) / 10.0
    t_norm = (temp - 20.0) / 10.0
    p_norm = (pH - 7.0) / 3.0
    
    # Quadratic terms
    # We have 10 params.
    # Let's allocate:
    # 0: Intercept
    # 1-4: Linear coefficients (b, s, t, p)
    # 5-8: Quadratic coefficients (b^2, s^2, t^2, p^2)
    # 9: Interaction term (b*s or t*p or a mix?)
    # Maybe a better interaction is s*t (substrate and temp often interact) or s*pH.
    # Let's use a generic interaction term: b*s + t*p? No, just one term to save params.
    # Let's try:
    # y = p0 + p1*b + p2*s + p3*t + p4*p + p5*b^2 + p6*s^2 + p7*t^2 + p8*p^2 + p9*(s*t)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    y = (p0 + 
         p1 * b_norm + 
         p2 * s_norm + 
         p3 * t_norm + 
         p4 * p_norm + 
         p5 * (b_norm ** 2) + 
         p6 * (s_norm ** 2) + 
         p7 * (t_norm ** 2) + 
         p8 * (p_norm ** 2) + 
         p9 * (s_norm * t_norm))
         
    return y

# EVOLVE-BLOCK-END
