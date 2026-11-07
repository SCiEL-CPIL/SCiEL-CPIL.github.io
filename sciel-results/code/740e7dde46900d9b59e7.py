"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a refined Ramberg-Osgood formulation:
    sigma = E(T) * eps + K(T) * sign(eps) * |eps|^n
    
    Where:
    - E(T) = p0 + p1*T + p2*T^2
    - K(T) = p3 + p4*T + p5*T^2
    - n    = p6
    
    Additional correction terms for flexibility:
    - p7 * eps^2
    - p8 * temp * eps
    - p9 * temp^2 * eps
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
    
    # Elastic modulus as quadratic in temperature
    E = p0 + p1 * temp + p2 * temp**2
    
    # Hardening coefficient as quadratic in temperature
    K = p3 + p4 * temp + p5 * temp**2
    
    # Hardening exponent - keep it positive for stability
    # Using abs + small offset ensures smoothness and positivity
    n = np.abs(p6) + 0.1
    
    # Base Ramberg-Osgood term
    abs_eps = np.abs(strain)
    sign_eps = np.sign(strain)
    # Handle zero strain carefully: sign(0)=0, so term is 0, which is correct for plastic part
    ro_term = E * strain + K * sign_eps * (abs_eps ** n)
    
    # Additional polynomial correction terms
    # These allow the model to fit deviations from the pure power-law hardening
    correction = (
        p7 * strain**2 +
        p8 * temp * strain +
        p9 * temp**2 * strain
    )
    
    return ro_term + correction
# EVOLVE-BLOCK-END
