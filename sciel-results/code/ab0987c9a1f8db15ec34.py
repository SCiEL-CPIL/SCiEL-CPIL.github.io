"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate using a biologically-grounded multiplicative model:
    - Monod kinetics for substrate saturation
    - Gaussian temperature and pH responses centered near physiological optima
    - Density-dependent inhibition (Haldane-type)
    
    Model: mu = mu_max * (s/(Ks+s)) * exp(-((T-Topt)^2)/(2*sigT^2)) * 
           exp(-((pH-pHopt)^2)/(2*sigpH^2)) * (1 - b/bmax)^n + offset
    """
    # Unpack 10 parameters
    mu_max = params[0]      # Maximum growth rate
    Ks = params[1]          # Monod half-saturation constant for substrate
    T_opt = params[2]       # Optimal temperature
    sigT = params[3]        # Temperature sensitivity (width)
    pH_opt = params[4]      # Optimal pH
    sigpH = params[5]       # pH sensitivity (width)
    b_max = params[6]       # Maximum density before inhibition
    n_inh = params[7]       # Inhibition exponent
    offset = params[8]      # Baseline offset
    scale = params[9]       # Additional scaling factor
    
    # Safety: ensure positive values for parameters that should be positive
    Ks_safe = np.abs(Ks) + 1e-10
    sigT_safe = np.abs(sigT) + 1e-10
    sigpH_safe = np.abs(sigpH) + 1e-10
    b_max_safe = np.abs(b_max) + 1e-10
    
    # Monod substrate term
    monod = s / (Ks_safe + s)
    
    # Gaussian temperature response
    temp_resp = np.exp(-((temp - T_opt) ** 2) / (2 * sigT_safe ** 2))
    
    # Gaussian pH response
    ph_resp = np.exp(-((pH - pH_opt) ** 2) / (2 * sigpH_safe ** 2))
    
    # Density-dependent inhibition: (1 - b/b_max)^n, clipped to be non-negative
    density_term = 1.0 - b / b_max_safe
    density_term = np.maximum(density_term, 0.0)
    density_inh = density_term ** n_inh
    
    # Combine all terms multiplicatively
    mu = mu_max * monod * temp_resp * ph_resp * density_inh * scale + offset
    
    return mu
# EVOLVE-BLOCK-END
