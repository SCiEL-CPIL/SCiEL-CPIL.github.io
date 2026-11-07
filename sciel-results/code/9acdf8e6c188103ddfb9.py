"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a refined Ramberg-Osgood formulation with normalized temperature
    and higher-order strain corrections to improve fit and conditioning.
    
    sigma = E(T) * eps + K(T) * sign(eps) * |eps|^n + C(T) * eps^2 + D * eps^3
    
    Where T is normalized temperature.
    """
    # Extract parameters
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
    
    # Normalize temperature to center around 0 for better polynomial conditioning
    # Training temp range is approx [0.06, 1.0]
    T = temp - 0.5
    
    # Elastic modulus as quadratic in normalized temperature
    E = p0 + p1 * T + p2 * T**2
    
    # Hardening coefficient as quadratic in normalized temperature
    K = p3 + p4 * T + p5 * T**2
    
    # Hardening exponent - keep it positive for stability
    n = np.abs(p6) + 0.1
    
    # Base Ramberg-Osgood term
    abs_eps = np.abs(strain)
    sign_eps = np.sign(strain)
    ro_term = E * strain + K * sign_eps * (abs_eps ** n)
    
    # Additional polynomial correction terms
    # p7 * eps^2 + p8 * eps^3 + p9 * T * eps^2
    # These capture non-linear elastic/plastic deviations
    correction = (
        p7 * strain**2 +
        p8 * strain**3 +
        p9 * T * strain**2
    )
    
    return ro_term + correction
# EVOLVE-BLOCK-END
