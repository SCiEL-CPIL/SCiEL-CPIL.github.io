import numpy as np
import time

def equation(b, s, temp, pH, params):
    """
    Generalized Additive Model (GAM) for E. coli growth rate.
    Uses linear combination of basis functions for each input variable.
    
    Inputs:
        b: biomass/density (range ~1-5)
        s: substrate concentration (range ~1-40)
        temp: temperature (range ~1-40)
        pH: pH level (range ~2-12)
    Params:
        p[0]: offset
        p[1]: coeff for temp basis 1 (Gaussian near low temp)
        p[2]: coeff for temp basis 2 (Gaussian near mid temp)
        p[3]: coeff for temp basis 3 (Gaussian near high temp)
        p[4]: coeff for pH basis 1 (Gaussian near low pH)
        p[5]: coeff for pH basis 2 (Gaussian near mid pH)
        p[6]: coeff for pH basis 3 (Gaussian near high pH)
        p[7]: coeff for substrate basis (Monod-like)
        p[8]: coeff for biomass basis (linear)
        p[9]: coeff for biomass basis (quadratic)
    """
    p = params
    
    # Normalize inputs to [0, 1] based on known ranges
    xb = (b - 1.0) / 4.0      # b: [1, 5]
    xs = (s - 1.0) / 39.0     # s: [1, 40]
    xT = (temp - 1.0) / 39.0  # T: [1, 40]
    xPH = (pH - 2.0) / 10.0   # pH: [2, 12]
    
    # Basis functions for temperature (3 Gaussians)
    # Centers at 0.25, 0.5, 0.75 on normalized scale
    T_b1 = np.exp(-0.5 * ((xT - 0.25) / 0.15)**2)
    T_b2 = np.exp(-0.5 * ((xT - 0.50) / 0.15)**2)
    T_b3 = np.exp(-0.5 * ((xT - 0.75) / 0.15)**2)
    
    # Basis functions for pH (3 Gaussians)
    # Centers at 0.25, 0.5, 0.75 on normalized scale
    pH_b1 = np.exp(-0.5 * ((xPH - 0.25) / 0.15)**2)
    pH_b2 = np.exp(-0.5 * ((xPH - 0.50) / 0.15)**2)
    pH_b3 = np.exp(-0.5 * ((xPH - 0.75) / 0.15)**2)
    
    # Substrate basis (Monod-like saturating function)
    s_b = xs / (0.3 + xs + 1e-9)
    
    # Biomass bases (linear and quadratic)
    b_b1 = xb
    b_b2 = xb**2
    
    # Linear combination of all basis functions
    result = (p[0] +
              p[1] * T_b1 + p[2] * T_b2 + p[3] * T_b3 +
              p[4] * pH_b1 + p[5] * pH_b2 + p[6] * pH_b3 +
              p[7] * s_b +
              p[8] * b_b1 + p[9] * b_b2)
    
    return result
