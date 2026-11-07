"""Stress-strain model for aluminium rod combining rational saturation with tanh transition."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Combines a rational saturation term (smooth elastic-plastic transition)
    with a tanh-based smooth step for additional shaping, plus polynomial
    correction terms for temperature and higher-order strain effects.
    
    Model structure:
    - Rational core: E(T)*strain / (1 + k(T)*strain) captures initial linear
      rise and natural saturation without explicit hinges
    - Tanh step: smooth transition that modulates the plastic region
    - Polynomial corrections: fine-tune curvature and temperature dependence
    
    Parameters:
        strain: array of strain values
        temp: array of temperature values
        params: array of 10 fitted constants
    
    Returns:
        array of predicted stress values
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure inputs are float64 arrays
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Temperature-dependent elastic modulus
    E = p0 + p1 * temp
    
    # Temperature-dependent saturation scale (controls how quickly stress plateaus)
    k = p2 + p3 * temp
    
    # Ensure k is positive to maintain proper saturation behavior
    k_safe = np.where(k > 0.01, k, 0.01)
    
    # Rational core: linear at low strain, saturates at high strain
    # For small strain: ~ E*strain
    # For large strain: ~ E/k (plateau)
    rational_core = E * strain / (1.0 + k_safe * strain)
    
    # Temperature-dependent yield strain threshold
    eps_y = p4 + p5 * temp
    
    # Transition width for the tanh step
    width = p6 + p7 * temp
    width_safe = np.maximum(width, 0.01)
    
    # Tanh smooth step: 0 well below eps_y, 1 well above eps_y
    x = (strain - eps_y) / width_safe
    step = 0.5 * (1.0 + np.tanh(x))
    
    # Modulate the rational core with the step to allow different
    # behavior before and after the yield region
    # Before yield: mostly the rational term (elastic)
    # After yield: blend toward a hardening term
    hardening_scale = 1.0 + p8 * temp
    
    # Plastic hardening component that adds stress beyond the rational plateau
    # This term is zero at strain = eps_y and grows smoothly
    eps_plastic = np.maximum(strain - eps_y, 0.0)
    hardening = hardening_scale * eps_plastic * np.exp(-k_safe * eps_plastic)
    
    # Final stress: rational core modulated by step + hardening contribution
    # At low strain (step~0): stress ~ rational_core (elastic)
    # At high strain (step~1): stress ~ rational_core + hardening (plastic)
    stress = rational_core * (1.0 - 0.3 * step) + hardening * step + p9
    
    return stress
# EVOLVE-BLOCK-END
