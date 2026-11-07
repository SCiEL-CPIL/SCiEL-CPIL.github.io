"""E. coli growth rate model with smooth Gaussian basis functions."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate modeled with smooth Gaussian basis functions.
    
    Uses Gaussian kernels in temperature and pH (smooth, differentiable) combined
    with Monod substrate saturation and Haldane density inhibition. This provides
    a smooth optimization landscape for BFGS while capturing the peaked growth
    response in temperature and pH.
    
    Basis functions:
      - Gaussian temperature response: exp(-((temp - Tc)^2) / (2*sigma_T^2))
      - Gaussian pH response: exp(-((pH - pHc)^2) / (2*sigma_pH^2))
      - Monod substrate saturation: s / (Ks + s)
      - Haldane density: b / (Kd + b + b^2/Ku)  (or 1/(1+Kd*b) for simplicity)
      - Interaction terms
    """
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Fixed biological parameters (smooth, no kinks)
    T_center = 37.0       # Optimal temperature
    T_sigma = 4.0         # Temperature Gaussian width
    pH_center = 7.0       # Optimal pH
    pH_sigma = 1.2        # pH Gaussian width
    Ks = 3.0              # Substrate half-saturation constant
    Kd = 0.5              # Density inhibition constant

    # Smooth Gaussian temperature response (always positive, peaks at T_center)
    temp_gauss = np.exp(-((temp - T_center) ** 2) / (2.0 * T_sigma ** 2))
    
    # Smooth Gaussian pH response (always positive, peaks at pH_center)
    ph_gauss = np.exp(-((pH - pH_center) ** 2) / (2.0 * pH_sigma ** 2))
    
    # Monod substrate saturation
    monod_s = s / (Ks + s)
    
    # Density inhibition (smooth, monotone decreasing)
    density_inhib = 1.0 / (1.0 + Kd * b)
    
    # Core interaction: all factors multiplied
    core = monod_s * temp_gauss * ph_gauss * density_inhib
    
    # Additive basis expansion with smooth functions
    prediction = (
        p0 * core +
        p1 * monod_s * temp_gauss +
        p2 * monod_s * ph_gauss +
        p3 * temp_gauss * ph_gauss +
        p4 * monod_s * density_inhib +
        p5 * temp_gauss +
        p6 * ph_gauss +
        p7 * monod_s +
        p8 * density_inhib * temp_gauss * ph_gauss +
        p9
    )
    
    return prediction
# EVOLVE-BLOCK-END
