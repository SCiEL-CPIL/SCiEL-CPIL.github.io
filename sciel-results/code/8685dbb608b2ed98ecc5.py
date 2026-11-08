# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model.
    
    Parameters:
    b: population density
    s: substrate concentration
    temp: temperature
    pH: pH level
    params: array of 10 parameters
        p[0]: mu_max (maximum growth rate)
        p[1]: K_s (Monod half-saturation constant)
        p[2]: T_opt (optimal temperature)
        p[3]: sigma_T (temperature width)
        p[4]: pH_opt (optimal pH)
        p[5]: sigma_pH (pH width)
        p[6]: K_b (density inhibition constant)
        p[7]: Monod exponent
        p[8]: density inhibition exponent
        p[9]: offset/baseline
    """
    p = params
    mu_max = p[0]
    K_s = np.abs(p[1]) + 1e-8
    T_opt = p[2]
    sigma_T = np.abs(p[3]) + 1e-8
    pH_opt = p[4]
    sigma_pH = np.abs(p[5]) + 1e-8
    K_b = np.abs(p[6]) + 1e-8
    monod_exp = np.abs(p[7]) + 0.1
    dens_exp = np.abs(p[8]) + 0.1
    offset = p[9]
    
    # Monod substrate kinetics
    monod = (s / (K_s + s)) ** monod_exp
    
    # Temperature bell curve
    temp_factor = np.exp(-((temp - T_opt) ** 2) / (2.0 * sigma_T ** 2))
    
    # pH bell curve
    ph_factor = np.exp(-((pH - pH_opt) ** 2) / (2.0 * sigma_pH ** 2))
    
    # Density inhibition (Logistic-like)
    dens_factor = 1.0 / (1.0 + (b / K_b) ** dens_exp)
    
    # Combine all factors
    mu = mu_max * monod * temp_factor * ph_factor * dens_factor + offset
    
    # Ensure finite output
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    return mu
# EVOLVE-BLOCK-END
