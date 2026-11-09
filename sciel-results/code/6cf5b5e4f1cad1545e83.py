"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature."""
    # params: 10 fitted constants
    # We model stress as a polynomial in strain with temperature-dependent coefficients.
    # This captures the non-linear stress-strain behavior of aluminium (elastic-plastic)
    # and the temperature softening effect.
    
    s = strain
    t = temp
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Temperature factor: reduces effective modulus/stress as temperature increases
    # t is in [0.06667, 1], so (1 - t) goes from ~0.933 to 0
    # We use a smooth factor that doesn't go negative
    temp_factor = 1.0 - p0 * (t - 0.06667) / (1.0 - 0.06667)
    # Ensure temp_factor is positive to avoid sign flips
    temp_factor = np.abs(temp_factor) + 1e-6
    
    # Polynomial in strain: captures elastic region (linear) and plastic region (non-linear)
    # s is in [-0.001911, 0.9636]
    # Use a combination of terms that can model the yield point and hardening
    
    # Base elastic-plastic model:
    # For small strain: linear (E*s)
    # For larger strain: power law or polynomial hardening
    
    # Let's use a polynomial that can approximate the stress-strain curve:
    # stress = (p1 + p2*t + p3*t^2) * (s + p4*s^2 + p5*s^3 + p6*s^4 + p7*s^5)
    # Plus a temperature offset term
    
    # More structured:
    # E_eff(t) = p1 + p2*t + p3*t^2
    # Hardening: f(s) = s + p4*s^2 + p5*s^3 + p6*s^4 + p7*s^5
    # Offset: p8*t + p9
    
    E_eff = p1 + p2 * t + p3 * t * t
    hardening = s + p4 * s * s + p5 * s * s * s + p6 * s * s * s * s + p7 * s * s * s * s * s
    
    stress = E_eff * hardening + p8 * t + p9
    
    return stress
# EVOLVE-BLOCK-END
