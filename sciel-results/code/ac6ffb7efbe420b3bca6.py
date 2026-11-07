"""E. coli growth rate symbolic regression with biologically-grounded model."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model with:
    - Monod substrate saturation
    - Gaussian temperature response (optimum ~37°C)
    - Gaussian pH response (optimum ~7.0)
    - Logistic density-dependent inhibition
    - Multiplicative combination for physical interpretability
    
    Parameters:
        b: population density (dimensionless, ~1-5)
        s: substrate concentration (~1-40)
        temp: temperature in Celsius (~1-5)
        pH: pH value (~12 in training? wait, pH range is [1, 4.999]... 
             Actually looking at inputs: [1, 4.998], [1, 4.999], [1.002, 39.98], [2, 12]
             So b in [1, 2], s in [1, 40], temp in [1, 5], pH in [12, 12]? 
             Wait, the input ranges are: [1, 4.998], [1, 4.999], [1.002, 39.98], [2, 12]
             That means: b in [1, 4.998], s in [1, 4.999], temp in [1.002, 39.98], pH in [2, 12]
    
    params: 10 fitted constants
    """
    # Unpack parameters
    p_max = params[0]       # max growth rate
    K_s = params[1]         # Monod half-saturation constant for substrate
    T_opt = params[2]       # optimal temperature
    T_width = params[3]     # temperature response width
    pH_opt = params[4]      # optimal pH
    pH_width = params[5]    # pH response width
    K_b = params[6]         # density inhibition constant
    n_inh = params[7]       # inhibition exponent
    offset = params[8]      # baseline offset
    scale = params[9]       # global scale factor

    # Ensure numerical stability: use softplus or clipping for positive parameters
    # K_s should be positive for Monod equation
    K_s_safe = np.abs(K_s) + 1e-6
    
    # Temperature and pH widths should be positive
    T_width_safe = np.abs(T_width) + 1e-6
    pH_width_safe = np.abs(pH_width) + 1e-6
    
    # K_b should be positive
    K_b_safe = np.abs(K_b) + 1e-6
    
    # Monod substrate term: s / (K_s + s)
    monod_s = s / (K_s_safe + s)
    
    # Gaussian temperature response
    temp_response = np.exp(-((temp - T_opt) ** 2) / (2.0 * T_width_safe ** 2))
    
    # Gaussian pH response
    ph_response = np.exp(-((pH - pH_opt) ** 2) / (2.0 * pH_width_safe ** 2))
    
    # Logistic density-dependent inhibition: 1 / (1 + (b/K_b)^n)
    # Use safe computation to avoid overflow
    ratio = b / K_b_safe
    # Clip ratio to avoid overflow in power
    ratio_clipped = np.clip(ratio, 0, 1e6)
    inhibition = 1.0 / (1.0 + ratio_clipped ** n_inh)
    
    # Combine multiplicatively
    growth = p_max * monod_s * temp_response * ph_response * inhibition
    
    # Apply scale and offset
    result = scale * growth + offset
    
    return result
# EVOLVE-BLOCK-END
