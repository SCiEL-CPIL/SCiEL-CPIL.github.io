import numpy as np
import time

_start_time = time.time()
_DEADLINE = _start_time + 140.0

def equation(b, s, temp, pH, params):
    p = params
    
    # Check deadline
    if time.time() > _DEADLINE:
        return np.zeros_like(b)
    
    # Normalize inputs to [-1, 1] based on training ranges
    # b: [1, 4.998]
    # s: [1, 4.999]
    # temp: [1.002, 39.98]
    # pH: [2, 12]
    
    b_norm = 2 * (b - 1.0) / (4.998 - 1.0) - 1
    s_norm = 2 * (s - 1.0) / (4.999 - 1.0) - 1
    temp_norm = 2 * (temp - 1.002) / (39.98 - 1.002) - 1
    ph_norm = 2 * (pH - 2.0) / (12.0 - 2.0) - 1
    
    # Clip to [-1, 1] to avoid extrapolation issues during fitting
    b_norm = np.clip(b_norm, -1, 1)
    s_norm = np.clip(s_norm, -1, 1)
    temp_norm = np.clip(temp_norm, -1, 1)
    ph_norm = np.clip(ph_norm, -1, 1)
    
    # Compute Legendre Polynomials up to order 4
    # L0 = 1
    # L1 = x
    # L2 = 0.5 * (3*x^2 - 1)
    # L3 = 0.5 * (5*x^3 - 3*x)
    # L4 = 0.375 * (35*x^4 - 30*x^2 + 3)
    
    # Temperature terms (up to 4th order)
    T_L1 = temp_norm
    T_L2 = 0.5 * (3 * temp_norm**2 - 1)
    T_L3 = 0.5 * (5 * temp_norm**3 - 3 * temp_norm)
    T_L4 = 0.375 * (35 * temp_norm**4 - 30 * temp_norm**2 + 3)
    
    # pH terms (up to 3rd order to save a slot for b/s if needed, or 4th?)
    # Let's try 4th for pH as well, dropping one linear term?
    # Current plan: T(1-4), pH(1-3), s(1), b(1) -> 4+3+1+1 = 9 terms + 1 const = 10 terms.
    
    pH_L1 = ph_norm
    pH_L2 = 0.5 * (3 * ph_norm**2 - 1)
    pH_L3 = 0.5 * (5 * ph_norm**3 - 3 * ph_norm)
    
    # Substrate and Biomass terms (1st order)
    s_L1 = s_norm
    b_L1 = b_norm
    
    # Strategy:
    # p[0]: constant
    # p[1]: T_L1
    # p[2]: T_L2
    # p[3]: T_L3
    # p[4]: T_L4
    # p[5]: pH_L1
    # p[6]: pH_L2
    # p[7]: pH_L3
    # p[8]: s_L1
    # p[9]: b_L1
    
    result = (p[0] 
              + p[1] * T_L1 
              + p[2] * T_L2 
              + p[3] * T_L3 
              + p[4] * T_L4 
              + p[5] * pH_L1 
              + p[6] * pH_L2 
              + p[7] * pH_L3 
              + p[8] * s_L1 
              + p[9] * b_L1)
              
    return result
