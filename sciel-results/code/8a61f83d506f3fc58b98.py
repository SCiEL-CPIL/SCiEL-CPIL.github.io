"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate modeled with a biologically-inspired multiplicative response surface.
    
    Structure:
    - Nutrients (b, s): Michaelis-Menten saturation via 1/(1+exp(-k*(x-K)))
    - Temperature: Gaussian-like optimal window
    - pH: Gaussian-like optimal window
    - Multiplicative combination with additive offset
    
    Parameters (10 total):
    p0: max growth rate scale
    p1: b saturation constant (K_b)
    p2: b half-saturation steepness
    p3: s saturation constant (K_s)  
    p4: s half-saturation steepness
    p5: temp optimum
    p6: temp width
    p7: pH optimum
    p8: pH width
    p9: baseline offset
    
    This captures:
    - Saturation in nutrients (biologically realistic)
    - Optimal temperature window (bell-shaped)
    - Optimal pH window (bell-shaped)
    - Non-negativity through sigmoid and Gaussian forms
    """
    p = params
    
    # Nutrient responses: sigmoid saturation
    # b: range [1, 5], s: range [1, 50]
    # Use sigmoid: 1/(1+exp(-steepness*(x - K)))
    # Clip parameters to avoid numerical issues
    K_b = np.clip(p[1], 0.5, 10.0)
    steep_b = np.clip(p[2], 0.1, 10.0)
    K_s = np.clip(p[3], 0.5, 100.0)
    steep_s = np.clip(p[4], 0.01, 10.0)
    
    b_resp = 1.0 / (1.0 + np.exp(-steep_b * (b - K_b)))
    s_resp = 1.0 / (1.0 + np.exp(-steep_s * (s - K_s)))
    
    # Temperature: Gaussian response
    # temp: range [1, 40], E. coli optimum ~37°C
    T_opt = np.clip(p[5], 5.0, 45.0)
    T_width = np.clip(p[6], 1.0, 20.0)
    temp_resp = np.exp(-0.5 * ((temp - T_opt) / T_width) ** 2)
    
    # pH: Gaussian response
    # pH: range [2, 12], E. coli optimum ~7
    pH_opt = np.clip(p[7], 3.0, 11.0)
    pH_width = np.clip(p[8], 0.5, 5.0)
    ph_resp = np.exp(-0.5 * ((pH - pH_opt) / pH_width) ** 2)
    
    # Multiplicative model with scale and offset
    scale = np.clip(p[0], 0.01, 10.0)
    offset = np.clip(p[9], -1.0, 1.0)
    
    return scale * b_resp * s_resp * temp_resp * ph_resp + offset
# EVOLVE-BLOCK-END
