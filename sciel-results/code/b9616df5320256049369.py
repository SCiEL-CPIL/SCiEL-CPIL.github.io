import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Ramberg-Osgood style stress-strain model with temperature-dependent coefficients.
    
    sigma = E(T) * eps + K(T) * (eps / eps_ref)^n(T)
    
    where:
      E(T)  = p0 + p1*T + p2*T^2
      K(T)  = p3 + p4*T + p5*T^2
      n(T)  = p6 + p7*T
      eps_ref = p8 (reference strain scale)
      p9: additional offset or correction term
    
    We use 10 parameters total.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    eps = strain
    T = temp
    
    # Ensure eps is non-negative for the power term (strain can be negative in training)
    eps_abs = np.abs(eps)
    
    # Temperature-dependent elastic modulus
    E = p0 + p1 * T + p2 * T * T
    
    # Temperature-dependent hardening coefficient
    K = p3 + p4 * T + p5 * T * T
    
    # Temperature-dependent hardening exponent
    n = p6 + p7 * T
    
    # Reference strain scale (avoid division by zero)
    eps_ref = np.where(np.abs(p8) < 1e-12, 1e-12, p8)
    
    # Ramberg-Osgood: sigma = E*eps + K*(eps/eps_ref)^n
    # For negative strain, the plastic term should also be negative (symmetric)
    # Use sign(eps) * (|eps|/eps_ref)^n
    if n > 0:
        plastic = K * np.sign(eps) * np.power(eps_abs / np.abs(eps_ref), n)
    else:
        # If n <= 0, fall back to linear only to avoid numerical issues
        plastic = np.zeros_like(eps)
    
    sigma = E * eps + plastic + p9
    
    return sigma
