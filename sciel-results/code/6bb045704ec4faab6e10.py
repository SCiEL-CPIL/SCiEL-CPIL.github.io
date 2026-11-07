import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for an aluminium rod using a modified Ramberg-Osgood formulation
    with temperature-dependent coefficients.
    
    sigma = E(T)*u + K(T) * (u)^n(T)
    
    Where:
      E(T) = p0 + p1*T          (Elastic modulus)
      K(T) = p2 + p3*T          (Hardening coefficient)
      n(T) = p4 + p5*T          (Hardening exponent)
    
    Parameters (10):
    p0, p1: Elastic modulus coefficients
    p2, p3: Hardening coefficient coefficients
    p4, p5: Hardening exponent coefficients
    p6, p7, p8, p9: Reserved for potential extension (not used in this version)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    u = strain
    T = temp
    
    # Temperature-dependent coefficients
    E = p0 + p1 * T
    K = p2 + p3 * T
    n = p4 + p5 * T
    
    # Ensure n is positive and not too extreme to avoid numerical issues
    n_safe = np.clip(n, 0.1, 10.0)
    
    # Ramberg-Osgood hardening term: K * u^n
    # Use absolute value for negative strains to handle compression
    u_abs = np.abs(u)
    # Clip to avoid 0^negative or very large values
    u_clipped = np.clip(u_abs, 1e-10, 10.0)
    roce_term = K * np.power(u_clipped, n_safe)
    # Preserve sign for negative strains
    roce_term = np.sign(u) * roce_term
    
    # Elastic term
    elastic = E * u
    
    # Combine all terms
    sigma = elastic + roce_term
    
    return sigma
