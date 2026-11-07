"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate from population density, substrate, temperature and pH.
    
    Uses a multivariate polynomial surface (up to 2nd order + 1 interaction) 
    in normalized inputs. This flexible form allows the optimizer to find 
    arbitrary optima and saturation behaviors without mechanistic constraints.
    
    Parameters (10 total):
    p[0]: Intercept
    p[1]: Coeff for b
    p[2]: Coeff for s
    p[3]: Coeff for temp
    p[4]: Coeff for pH
    p[5]: Coeff for b^2
    p[6]: Coeff for s^2
    p[7]: Coeff for temp^2
    p[8]: Coeff for pH^2
    p[9]: Coeff for temp * pH interaction
    """
    p = params
    
    # Normalize inputs to approximately [-1, 1] for better conditioning
    # b: [1, 5] -> center 3, scale 2
    # s: [1, 5] -> center 3, scale 2
    # temp: [1, 40] -> center 20.5, scale 19.5
    # pH: [2, 12] -> center 7, scale 5
    
    b_norm = (b - 3.0) / 2.0
    s_norm = (s - 3.0) / 2.0
    t_norm = (temp - 20.5) / 19.5
    ph_norm = (pH - 7.0) / 5.0
    
    # Polynomial terms
    # Linear
    term_b = b_norm
    term_s = s_norm
    term_t = t_norm
    term_ph = ph_norm
    
    # Quadratic
    term_b2 = b_norm * b_norm
    term_s2 = s_norm * s_norm
    term_t2 = t_norm * t_norm
    term_ph2 = ph_norm * ph_norm
    
    # Interaction
    term_t_ph = t_norm * ph_norm
    
    # Combine
    result = (p[0] 
              + p[1] * term_b 
              + p[2] * term_s 
              + p[3] * term_t 
              + p[4] * term_ph 
              + p[5] * term_b2 
              + p[6] * term_s2 
              + p[7] * term_t2 
              + p[8] * term_ph2 
              + p[9] * term_t_ph)
              
    return result
# EVOLVE-BLOCK-END
