"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a Ramberg-Osgood formulation with a cubic correction term:
    sigma = E(T) * eps + K(T) * sign(eps) * |eps|^n + C(T) * eps^3
    
    Where:
    - E(T) = p0 + p1*T + p2*T^2  (Elastic Modulus)
    - K(T) = p3 + p4*T + p5*T^2  (Hardening Coefficient)
    - n    = p6                  (Hardening Exponent)
    - C(T) = p7 + p8*T + p9*T^2  (Cubic Correction Coefficient)
    """
    # Extract parameters
    p0, p1, p2 = params[0], params[1], params[2]
    p3, p4, p5 = params[3], params[4], params[5]
    p6 = params[6]
    p7, p8, p9 = params[7], params[8], params[9]
    
    # Elastic modulus as quadratic in temperature
    E = p0 + p1 * temp + p2 * temp**2
    
    # Hardening coefficient as quadratic in temperature
    K = p3 + p4 * temp + p5 * temp**2
    
    # Hardening exponent - keep it positive for stability
    # Using abs + small offset ensures smoothness and positivity
    n = np.abs(p6) + 0.1
    
    # Cubic correction coefficient as quadratic in temperature
    C = p7 + p8 * temp + p9 * temp**2
    
    # Base Ramberg-Osgood term
    abs_eps = np.abs(strain)
    sign_eps = np.sign(strain)
    
    # Plastic/Hardening term
    # Using sign(eps) * |eps|^n ensures correct sign for negative strain
    # and handles the non-integer exponent safely.
    hardening_term = K * sign_eps * (abs_eps ** n)
    
    # Elastic term
    elastic_term = E * strain
    
    # Cubic correction term
    # This term helps capture deviations from the pure power law,
    # particularly near the yield point or at higher strains where
    # the curve might flatten or steepen differently than eps^n.
    cubic_term = C * strain**3
    
    return elastic_term + hardening_term + cubic_term
# EVOLVE-BLOCK-END
