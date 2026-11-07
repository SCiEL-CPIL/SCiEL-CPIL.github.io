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
    
    # Additive polynomial model with quadratic terms and key interactions
    # This gives the optimizer maximum flexibility without structural constraints
    
    # Linear terms
    term_b_lin = p[0] * b_norm
    term_s_lin = p[1] * s_norm
    term_t_lin = p[2] * t_norm
    term_ph_lin = p[3] * ph_norm
    
    # Quadratic terms
    term_b_quad = p[4] * b_norm**2
    term_s_quad = p[5] * s_norm**2
    term_t_quad = p[6] * t_norm**2
    term_ph_quad = p[7] * ph_norm**2
    
    # Key interaction: temperature * pH (biologically plausible)
    term_interaction = p[8] * t_norm * ph_norm
    
    # Offset
    offset = p[9]
    
    y = offset + term_b_lin + term_s_lin + term_t_lin + term_ph_lin + \
         term_b_quad + term_s_quad + term_t_quad + term_ph_quad + \
         term_interaction
    
    # Ensure finite output
    if not np.all(np.isfinite(y)):
        return np.zeros_like(y)
        
    return y
