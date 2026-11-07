import numpy as np
import time

_start_time = time.time()
_DEADLINE = _start_time + 140.0

def equation(b, s, temp, pH, params):
    p = params
    
    # Substrate: Monod kinetics with Hill coefficient
    # Use stable formulation: 1 / (1 + (ks/s)^hill)
    ks = np.exp(p[1])
    hill = np.exp(p[9])
    # Avoid division by zero: s is in [1, 4.999] so s > 0
    ratio = ks / np.clip(s, 1e-10, None)
    monod = 1.0 / (1.0 + ratio ** hill)
    
    # Temperature: Gaussian centered at optimal
    t_opt = p[2]
    t_width = np.exp(p[3])
    temp_factor = np.exp(-0.5 * ((temp - t_opt) / t_width) ** 2)
    
    # pH: Gaussian centered at optimal
    ph_opt = p[4]
    ph_width = np.exp(p[5])
    ph_factor = np.exp(-0.5 * ((pH - ph_opt) / ph_width) ** 2)
    
    # Buffer/biomass effect: linear + quadratic
    b_effect = p[8] + p[6] * b + p[7] * b ** 2
    
    # Global scale
    result = p[0] * monod * temp_factor * ph_factor * b_effect
    
    return result
