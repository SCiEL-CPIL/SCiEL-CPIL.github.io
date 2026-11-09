"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """E. coli growth rate using a polynomial basis model.
    
    Uses a 10-term polynomial basis with biologically-motivated terms:
    linear, quadratic, and key interaction terms. Linear in parameters,
    so BFGS optimization is convex.
    """
    b = np.asarray(b, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    pH = np.asarray(pH, dtype=np.float64)
    
    # 10 basis functions
    phi0 = np.ones_like(b)       # bias
    phi1 = b                     # linear: biomass
    phi2 = s                     # linear: substrate
    phi3 = temp                  # linear: temperature
    phi4 = pH                    # linear: pH
    phi5 = b * s                 # interaction: biomass-substrate
    phi6 = temp * temp           # quadratic: temperature (Gaussian-like)
    phi7 = pH * pH               # quadratic: pH
    phi8 = b * temp              # interaction: biomass-temp
    phi9 = s * temp              # interaction: substrate-temp
    
    # Linear combination
    out = (params[0] * phi0 +
           params[1] * phi1 +
           params[2] * phi2 +
           params[3] * phi3 +
           params[4] * phi4 +
           params[5] * phi5 +
           params[6] * phi6 +
           params[7] * phi7 +
           params[8] * phi8 +
           params[9] * phi9)
    
    return np.where(np.isfinite(out), out, 0.0)
# EVOLVE-BLOCK-END
