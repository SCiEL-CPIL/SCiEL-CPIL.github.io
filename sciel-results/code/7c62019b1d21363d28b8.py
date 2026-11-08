"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate: modified Monod with Arrhenius temperature,
    quadratic pH response, and logistic density inhibition.
    
    mu = mu_max * (s / (K_s + s)) * f_T * f_pH * f_b + offset
    
    f_T: Arrhenius-type temperature factor
    f_pH: quadratic (bell-shaped) pH factor
    f_b: logistic density inhibition
    """
    # Extract parameters with safe clipping for stability
    mu_max = np.clip(params[0], 1e-6, 1e6)       # max growth rate
    K_s = np.abs(params[1]) + 1e-6                # Monod half-saturation
    T_opt = params[2]                              # optimal temperature
    E_act = np.abs(params[3]) + 1e-6              # activation energy scale
    pH_opt = params[4]                             # optimal pH
    pH_width = np.abs(params[5]) + 0.1             # pH bell width
    K_b = np.abs(params[6]) + 1e-6                # density inhibition constant
    b_max = np.abs(params[7]) + 1.0               # carrying capacity scale
    T_scale = np.abs(params[8]) + 1.0             # temperature sensitivity
    offset = params[9]                             # baseline offset

    # Monod substrate saturation
    monod = s / (K_s + s)

    # Arrhenius-type temperature factor: peaks at T_opt
    # f_T = exp(-E_act * (T - T_opt)^2 / (T * T_scale))
    T_diff = temp - T_opt
    f_T = np.exp(-E_act * T_diff ** 2 / (np.abs(temp) + 1e-6) / T_scale)

    # Quadratic pH bell: f_pH = exp(-(pH - pH_opt)^2 / (2 * pH_width^2))
    pH_diff = pH - pH_opt
    f_pH = np.exp(-(pH_diff ** 2) / (2.0 * pH_width ** 2))

    # Logistic density inhibition: f_b = 1 / (1 + (b / K_b)^(b_max))
    # Use stable computation
    ratio = np.abs(b) / K_b
    # Clip to avoid overflow in power
    ratio_clipped = np.clip(ratio, 1e-10, 1e10)
    f_b = 1.0 / (1.0 + ratio_clipped ** b_max)

    # Combine
    mu = mu_max * monod * f_T * f_pH * f_b + offset

    # Ensure finite output
    mu = np.where(np.isfinite(mu), mu, 0.0)

    return mu
# EVOLVE-BLOCK-END
