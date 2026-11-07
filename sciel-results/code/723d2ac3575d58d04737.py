"""E. coli growth rate model with flexible basis expansion."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate modeled with a flexible additive basis expansion.
    
    Uses fixed biological centers and explicit interaction terms to reduce
    non-convexity for BFGS optimization while capturing non-separable effects.
    
    Basis functions:
      - Monod substrate saturation: s / (Ks + s)
      - Gaussian temperature optimum
      - Gaussian pH optimum
      - Density inhibition: 1 / (1 + Kd * b)
      - Interaction terms between key variables
      - Polynomial corrections
    """
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Fixed biological centers (based on E. coli physiology)
    Ks = 5.0        # Substrate half-saturation
    T_opt = 37.0    # Optimal temperature
    T_sig = 5.0     # Temperature sensitivity
    pH_opt = 7.0    # Optimal pH
    pH_sig = 1.0    # pH sensitivity
    Kd = 1.0        # Density inhibition constant
    
    # Core basis functions
    monod_s = s / (Ks + s)
    gauss_temp = np.exp(-((temp - T_opt) ** 2) / (2.0 * T_sig ** 2))
    gauss_ph = np.exp(-((pH - pH_opt) ** 2) / (2.0 * pH_sig ** 2))
    density_inhib = 1.0 / (1.0 + Kd * b)
    
    # Interaction terms
    temp_ph_interaction = gauss_temp * gauss_ph
    substrate_temp_interaction = monod_s * gauss_temp
    substrate_ph_interaction = monod_s * gauss_ph
    density_substrate_interaction = density_inhib * monod_s
    
    # Additional flexible terms
    linear_s = s
    linear_temp = temp
    linear_ph = pH
    
    # Combine all basis functions with fitted parameters
    prediction = (
        p0 * monod_s * gauss_temp * gauss_ph * density_inhib +
        p1 * monod_s * gauss_temp +
        p2 * monod_s * gauss_ph +
        p3 * gauss_temp * gauss_ph +
        p4 * monod_s +
        p5 * gauss_temp +
        p6 * gauss_ph +
        p7 * density_inhib * monod_s +
        p8 * linear_s +
        p9
    )
    
    return prediction
# EVOLVE-BLOCK-END
