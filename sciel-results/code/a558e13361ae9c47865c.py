import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for an aluminium rod using a Ramberg-Osgood formulation
    with temperature-dependent parameters.
    
    sigma = E(T) * epsilon + K(T) * (epsilon + c(T))^n(T)
    
    Parameters (10):
    p0, p1, p2: E(T) = p0 + p1*T + p2*T^2 (elastic modulus)
    p3, p4: K(T) = p3 + p4*T (hardening coefficient)
    p5, p6: c(T) = p5 + p6*T (offset for power law)
    p7, p8: n(T) = p7 + p8*T (hardening exponent)
    p9: correction factor
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    u = strain
    T = temp
    
    # Temperature-dependent coefficients
    E = p0 + p1 * T + p2 * T * T
    K = p3 + p4 * T
    c = p5 + p6 * T
    n = p7 + p8 * T
    
    # Ensure n is positive for stable power computation
    n_safe = np.maximum(n, 0.1)
    
    # Ramberg-Osgood form with offset
    # The offset c helps capture the transition from elastic to plastic
    # We use (u + c) where c can be negative to shift the curve
    base = u + c
    # Ensure base is positive for real-valued power
    base_safe = np.maximum(base, 1e-8)
    
    # Power law term
    power_term = np.power(base_safe, n_safe)
    
    # Combine elastic and plastic contributions
    sigma = E * u + K * power_term
    
    # Apply correction factor
    sigma = sigma * (1.0 + p9 * u * T)
    
    return sigma
