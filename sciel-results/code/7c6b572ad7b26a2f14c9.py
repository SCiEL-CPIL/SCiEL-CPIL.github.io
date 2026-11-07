import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Ramberg-Osgood model for stress-strain of an aluminium rod.
    
    sigma = E(T) * strain + K(T) * |strain|^n(T) * sign(strain)
    
    Where:
    E(T) = p0 + p1*T + p2*T^2       (Elastic modulus)
    K(T) = p3 + p4*T + p5*T^2       (Hardening coefficient)
    n(T) = p6 + p7*T + p8*T^2       (Hardening exponent)
    p9:  Additional offset for stress (to capture any initial stress)
    
    The sign function is smoothed using a tanh approximation for differentiability.
    """
    p = params
    
    # Safety check for non-finite parameters
    if not np.all(np.isfinite(p)):
        return np.zeros_like(strain)
    
    T = temp
    
    # Temperature-dependent parameters
    E = p[0] + p[1] * T + p[2] * T * T
    K = p[3] + p[4] * T + p[5] * T * T
    n = p[6] + p[7] * T + p[8] * T * T
    
    # Offset
    offset = p[9]
    
    # Ensure n is positive to avoid numerical issues
    n_safe = np.maximum(n, 0.1)
    
    # Compute |strain|^n * sign(strain) using smooth approximation
    # For differentiability, we use: sign(x) * |x|^n = x * |x|^(n-1)
    # And |x| = sqrt(x^2) which is smooth except at 0, but x*|x|^(n-1) is smooth for n>1
    # For n in [0.1, ...], we need a smooth approximation
    
    # Use: f(x) = x * (x^2 + eps)^(n/2 - 1) * (x^2 + eps)^(1/2) / (x^2 + eps)^(1/2)
    # Simpler: use the identity x*|x|^(n-1) = x * (x^2)^((n-1)/2)
    # To make it smooth at 0, use: x * (x^2 + delta)^((n-1)/2)
    # where delta is a small constant
    
    delta = 1e-6
    abs_strain_pow = np.power(strain * strain + delta, (n_safe - 1.0) / 2.0)
    nonlinear = strain * abs_strain_pow
    
    sigma = E * strain + K * nonlinear + offset
    
    return sigma
