"""Structural change: multiplicative temperature-modulated elastic-plastic model."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Multiplicative temperature-modulated elastic-plastic constitutive model.
    
    Structure:
      E(T)   = p0 * (1 + p1*T + p2*T^2)          # temp-dependent elastic modulus
      sigma_y(T) = p3 * (1 + p4*T)                # temp-dependent yield stress
      hardening = sigma_y(T) * sinh(p5*eps) / (1 + p6*sinh(p5*eps)^2)  # saturation hardening
      stress = E(T) * eps * (1 + p7*eps + p8*eps^2) + hardening * (1 + p9*eps)
    
    This captures the physical interplay between temperature and plasticity
    through multiplicative coupling rather than additive basis functions.
    """
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    eps = strain
    T = temp
    
    # Temperature-dependent elastic modulus
    E_T = p0 * (1.0 + p1 * T + p2 * T * T)
    
    # Temperature-dependent yield stress
    sigma_y = p3 * (1.0 + p4 * T)
    
    # Smooth hardening with saturation via sinh
    # sinh(p5*eps) grows linearly for small eps, saturates via denominator
    s = p5 * eps
    sinh_s = np.sinh(s)
    denom = 1.0 + p6 * sinh_s * sinh_s
    hardening = sigma_y * sinh_s / denom
    
    # Elastic part with slight nonlinearity
    elastic = E_T * eps * (1.0 + p7 * eps + p8 * eps * eps)
    
    # Total stress
    stress = elastic + hardening * (1.0 + p9 * eps)
    
    return stress
# EVOLVE-BLOCK-END
