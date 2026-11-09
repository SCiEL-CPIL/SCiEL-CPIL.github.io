import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for aluminium rod.
    Uses a linear elastic term and a non-linear power-law hardening term,
    modulated by temperature.
    """
    eps = strain
    T = temp
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure numerical stability for the power term
    abs_eps = np.abs(eps) + 1e-8
    sign_eps = np.sign(eps)
    
    # Exponent for the hardening term
    n = p4 + p5 * T
    
    # Hardening term: H * sign(eps) * |eps|^n
    # If n is very large, this might overflow? 
    # Max eps ~ 1. 1^large = 1.
    # Min eps ~ 0. 0^large = 0.
    # If n is negative, 0^neg = inf. 
    # The 1e-8 prevents inf, but 1e-8^-10 is 1e80. Still finite.
    
    hardening_term = (p2 + p3 * T) * sign_eps * np.power(abs_eps, n)
    
    # Elastic term
    elastic_term = (p0 + p1 * T) * eps
    
    # Bias and interaction terms
    bias_term = p6 + p7 * T
    
    # Additional flexibility terms
    extra_term = p8 * eps * T + p9 * eps * eps
    
    return elastic_term + hardening_term + bias_term + extra_term
