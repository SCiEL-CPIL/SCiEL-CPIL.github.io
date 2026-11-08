import numpy as np
def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    pH = np.asarray(pH, dtype=np.float64)
    
    mu_max = params[0]
    Ks = params[1]
    Topt = params[2]
    aT = params[3]
    pHopt = params[4]
    aP = params[5]
    Kd = params[6]
    offset = params[7]
    w_st = params[8]
    w_pt = params[9]
    
    # Safe divisions
    Ks_safe = np.clip(Ks, 1e-6, 1e6)
    aT_safe = np.clip(aT, 1e-6, 1e6)
    aP_safe = np.clip(aP, 1e-6, 1e6)
    Kd_safe = np.clip(Kd, 1e-6, 1e6)
    
    # Haldane for substrate: s / (Ks + s + s^2/Ki)
    # We don't have a Ki parameter explicitly, but we can approximate inhibition 
    # or just use Monod if s doesn't inhibit. 
    # Given s range [1, 40], Monod s/(Ks+s) is safe. 
    # Let's use Monod for S to save a parameter for interactions.
    monod_s = s / (Ks_safe + s)
    
    # Rational bell for Temperature: 1 / (1 + aT * (T - Topt)^2)
    dT = temp - Topt
    temp_factor = 1.0 / (1.0 + aT_safe * dT * dT)
    
    # Rational bell for pH: 1 / (1 + aP * (pH - pHopt)^2)
    dP = pH - pHopt
    ph_factor = 1.0 / (1.0 + aP_safe * dP * dP)
    
    # Density inhibition: 1 / (1 + (b/Kd)^2)
    density_factor = 1.0 / (1.0 + (b / Kd_safe) ** 2)
    
    # Base term
    growth = mu_max * monod_s * temp_factor * ph_factor * density_factor
    
    # Interactions to allow more flexibility
    # w_st * monod_s * temp_factor
    # w_pt * ph_factor * temp_factor
    growth += w_st * monod_s * temp_factor
    growth += w_pt * ph_factor * temp_factor
    
    growth += offset
    
    return growth
