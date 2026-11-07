"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Model:
    stress = E(T) * strain + A(T) * (|strain| + eps0)**n(T) * sign(strain) + C(T)
    where:
      E(T) = p0 + p1*T
      A(T) = p2 + p3*T
      n(T) = max(p4 + p5*T, 0.1)
      C(T) = p7*T + p8*T^2 + p9
    
    This is a Ramberg-Osgood type model with temperature-dependent coefficients.
    The power-law hardening term captures non-linear stress-strain behavior,
    while temperature-dependent coefficients capture thermal softening.
    
    Params:
    0: p0 - base elastic modulus
    1: p1 - temperature coefficient of modulus
    2: p2 - base hardening coefficient
    3: p3 - temperature coefficient of hardening
    4: p4 - base hardening exponent
    5: p5 - temperature coefficient of exponent
    6: p6 - offset strain eps0 (small positive to avoid singularity)
    7: p7 - linear temp correction
    8: p8 - quadratic temp correction
    9: p9 - constant offset
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Temperature-dependent elastic modulus
    E = p0 + p1 * temp
    
    # Temperature-dependent hardening coefficient
    A = p2 + p3 * temp
    
    # Temperature-dependent hardening exponent (clamped to avoid issues)
    n = p4 + p5 * temp
    n = np.maximum(n, 0.1)
    
    # Strain term with small offset to avoid issues at zero
    eps0 = abs(p6) + 1e-6
    abs_strain = np.abs(strain)
    sign_strain = np.sign(strain)
    
    # Plastic/hardening term (power-law)
    hardening_term = A * (abs_strain + eps0) ** n * sign_strain
    
    # Elastic term
    elastic_term = E * strain
    
    # Additional temperature-dependent corrections
    temp_correction = p7 * temp + p8 * temp * temp + p9
    
    # Total stress
    stress = elastic_term + hardening_term + temp_correction
    
    return stress
# EVOLVE-BLOCK-END
