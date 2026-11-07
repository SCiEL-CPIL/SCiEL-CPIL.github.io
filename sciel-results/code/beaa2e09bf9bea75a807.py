"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a Ramberg-Osgood formulation:
    sigma = E(T) * eps + K(T) * sign(eps) * |eps|^n(T)
    
    Where:
    - E(T) = p0 + p1*T          (elastic modulus)
    - K(T) = p2 + p3*T          (hardening coefficient)
    - n(T) = p4 + p5*T          (hardening exponent, kept > 1)
    
    Plus polynomial correction terms for residual fitting:
    - p6*eps^2 + p7*eps^3 + p8*T*eps + p9*T^2*eps
    """
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    # Elastic modulus as linear function of temperature
    E = p0 + p1 * temp
    
    # Hardening coefficient as linear function of temperature
    K = p2 + p3 * temp
    
    # Hardening exponent - ensure it's > 1 for smooth behavior
    n = np.abs(p4 + p5 * temp) + 1.5
    
    # Ramberg-Osgood term
    abs_eps = np.abs(strain)
    sign_eps = np.sign(strain)
    ro_term = E * strain + K * sign_eps * (abs_eps ** n)
    
    # Polynomial correction terms
    correction = (
        p6 * strain**2 +
        p7 * strain**3 +
        p8 * temp * strain +
        p9 * temp**2 * strain
    )
    
    return ro_term + correction
# EVOLVE-BLOCK-END
