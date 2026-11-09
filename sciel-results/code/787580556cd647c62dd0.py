"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a flexible basis with multiple smooth nonlinear functions to capture
    elastic-plastic transition, work hardening, and temperature effects.
    
    params (10 constants):
      p0: Intercept
      p1: Linear strain coeff
      p2: Quadratic strain coeff
      p3: Quartic strain coeff
      p4: Temperature linear coeff
      p5: Temperature-strain interaction
      p6: Saturation amplitude
      p7: Saturation rate
      p8: Temperature-modulated saturation rate
      p9: Additional nonlinear amplitude
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    eps = strain
    T = temp
    
    # Polynomial basis in strain (up to degree 4)
    eps2 = eps * eps
    eps3 = eps2 * eps
    eps4 = eps2 * eps2
    
    # Temperature basis
    T2 = T * T
    
    # Non-linear saturation terms with different profiles
    # tanh: smooth S-curve for yield transition
    rate1 = p7
    sat_tanh = np.tanh(rate1 * eps)
    
    # arctan: different saturation profile, more gradual
    rate2 = p8 + T  # temperature modulates the rate
    sat_arctan = np.arctan(rate2 * eps)
    
    # softplus-like: max(0, x) smooth approximation for plastic region
    # Using np.logaddexp(0, x) which is softplus
    rate3 = p7
    sat_softplus = np.logaddexp(0.0, rate3 * eps)
    
    # Construct the full model
    sigma = p0 + p1 * eps + p2 * eps2 + p3 * eps4
    sigma += p4 * T + p5 * T * eps
    sigma += p6 * sat_tanh
    sigma += p9 * sat_arctan
    sigma += p9 * sat_softplus  # Reuse p9 for both additional nonlinear terms
    
    return sigma
# EVOLVE-BLOCK-END
