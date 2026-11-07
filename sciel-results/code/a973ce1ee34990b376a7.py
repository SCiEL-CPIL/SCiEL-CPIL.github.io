import numpy as np
import time

_start_time = time.time()
_DEADLINE = _start_time + 140.0

def equation(b, s, temp, pH, params):
    p = params
    
    # Substrate: Monod kinetics with Hill coefficient
    ks = np.exp(p[1])
    hill = np.exp(p[9])
    monod = (s ** hill) / (ks ** hill + s ** hill)
    
    # Temperature: Gaussian centered at optimal
    t_opt = p[2]
    t_width = np.exp(p[3])
    temp_factor = np.exp(-0.5 * ((temp - t_opt) / t_width) ** 2)
    
    # pH: Gaussian centered at optimal
    ph_opt = p[4]
    ph_width = np.exp(p[5])
    ph_factor = np.exp(-0.5 * ((pH - ph_opt) / ph_width) ** 2)
    
    # New structural change: Additive interaction terms instead of pure multiplicative
    # p[6] = coefficient for b*monod interaction
    # p[7] = coefficient for temp_factor*pH interaction  
    # p[8] = offset
    
    base = p[0] * monod * temp_factor * ph_factor
    interaction_b = p[6] * b * monod
    interaction_tp = p[7] * temp_factor * ph_factor
    offset = p[8]
    
    result = base + interaction_b + interaction_tp + offset
    
    return result
