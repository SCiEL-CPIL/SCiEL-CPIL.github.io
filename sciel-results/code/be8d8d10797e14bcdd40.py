"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a physics-informed model:
    - Elastic region: linear stress-strain with temperature-dependent Young's modulus
    - Plastic region: Ramberg-Osgood type hardening with saturation
    - Temperature effects on yield strength and hardening
    
    Model:
    sigma = E(T) * eps for eps < eps_y(T)
    sigma = sigma_y(T) + K(T) * (eps - eps_y(T))^n for eps >= eps_y(T)
    
    With smooth transition using a sigmoid-like function to avoid kinks.
    
    Parameters:
    p0: Young's modulus scale
    p1: Temperature coefficient for Young's modulus
    p2: Yield strain base value
    p3: Temperature coefficient for yield strain
    p4: Yield stress scale
    p5: Temperature coefficient for yield stress
    p6: Hardening modulus scale
    p7: Temperature coefficient for hardening modulus
    p8: Hardening exponent
    p9: Saturation stress scale
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Temperature-dependent coefficients
    E = p0 + p1 * temp  # Young's modulus
    eps_y = p2 + p3 * temp  # Yield strain
    sigma_y = p4 + p5 * temp  # Yield stress
    K = p6 + p7 * temp  # Hardening modulus
    
    # Ensure positive values where physically meaningful
    E = np.maximum(E, 1e-10)
    K = np.maximum(K, 1e-10)
    
    # Plastic strain
    eps_p = np.maximum(strain - eps_y, 0)
    
    # Hardening term with saturation: K * eps_p^p8 / (1 + eps_p^p8)
    # This gives a smooth saturation to p9
    n = p8
    # Handle n=0 case
    if abs(n) < 1e-10:
        hardening = K * eps_p
    else:
        eps_p_safe = np.maximum(eps_p, 1e-15)
        hardening = K * eps_p_safe**n / (1 + eps_p_safe**n)
    
    # Elastic stress
    sigma_elastic = E * strain
    
    # Smooth transition between elastic and plastic
    # Use a smoothstep-like function
    transition_width = 0.001
    t = (strain - eps_y) / transition_width
    # Smooth sigmoid: 0.5 * (1 + tanh(t))
    blend = 0.5 * (1 + np.tanh(t * 10))
    
    # Plastic stress
    sigma_plastic = sigma_y + hardening
    
    # Combine with smooth blending
    sigma = (1 - blend) * sigma_elastic + blend * sigma_plastic
    
    # Add saturation limit
    sigma_max = p9
    sigma = sigma * (1 - np.exp(-sigma / (sigma_max + 1e-10))) / (1 - np.exp(-1))
    
    return sigma
# EVOLVE-BLOCK-END
