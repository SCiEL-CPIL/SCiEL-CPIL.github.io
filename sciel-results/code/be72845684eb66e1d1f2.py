# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature.
    
    params (10 constants):
      p0: elastic modulus scale
      p1: yield strain
      p2: yield stress scale
      p3: hardening exponent
      p4: hardening stress scale
      p5: temperature softening coefficient
      p6: temperature effect on modulus
      p7: offset/intercept
      p8: compression stiffness modifier
      p9: saturation strain for hardening
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Temperature-dependent modulus: E(T) = p0 * (1 - p6 * temp)
    E = p0 * (1.0 - p6 * temp)
    
    # Temperature-dependent yield stress: sigma_y(T) = p2 * (1 - p5 * temp)
    sigma_y = p2 * (1.0 - p5 * temp)
    
    # Handle compression vs tension
    # For compression (strain < 0), use a modified stiffness
    comp_mask = strain < 0
    tens_mask = strain >= 0
    
    # Tension: elastic-plastic with hardening
    # Elastic: sigma = E * strain
    # Plastic: sigma = sigma_y + p4 * (1 - exp(-p3 * (strain - p1))) / (1 - exp(-p3 * p9))
    # Smooth transition using a soft hinge
    
    eps = strain  # tensile strain
    
    # Smooth yield: use a tanh-based smooth hinge
    # sigma_plastic = sigma_y + p4 * (1 - exp(-p3 * max(eps - p1, 0)))
    # To make it smooth, use: max(eps - p1, 0) approximated by (eps - p1 + sqrt((eps-p1)^2 + eps^2)) / 2
    excess = eps - p1
    # Smooth max: smooth_max(x, 0) = (x + sqrt(x^2 + eps_smooth)) / 2
    eps_smooth = 1e-4
    smooth_excess = (excess + np.sqrt(excess**2 + eps_smooth)) / 2.0
    
    # Hardening term with saturation
    hardening = p4 * (1.0 - np.exp(-p3 * smooth_excess))
    
    # Elastic part
    elastic = E * eps
    
    # Total tensile stress: elastic in elastic region, yield + hardening in plastic
    # Use smooth blending
    # When excess is small (elastic), stress ≈ elastic
    # When excess is large (plastic), stress ≈ sigma_y + hardening
    # Blend factor: how far into plastic we are
    blend = smooth_excess / (smooth_excess + 1e-3)
    
    sigma_tens = elastic * (1 - blend) + (sigma_y + hardening) * blend
    
    # Compression: linear with modified stiffness
    sigma_comp = E * p8 * strain
    
    # Combine
    sigma = np.where(comp_mask, sigma_comp, sigma_tens)
    
    # Add offset
    sigma = sigma + p7
    
    return sigma
# EVOLVE-BLOCK-END
