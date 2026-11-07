import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Pure Ramberg-Osgood model for stress-strain of an aluminium rod.
    
    sigma = E(T) * strain + K(T) * |strain|^n * sign(strain) + C(T)
    
    where:
    E(T) = p0 + p1*T  (elastic modulus, linear in temperature)
    K(T) = p2 + p3*T  (hardening coefficient, linear in temperature)
    n    = p4         (hardening exponent, constant)
    C(T) = p5 + p6*T  (offset/intercept, linear in temperature)
    
    Uses smooth sign function for differentiability.
    The elastic term captures the initial linear regime exactly.
    The power-law term captures nonlinear hardening.
    The offset allows for any constant shift.
    """
    p = params
    
    strain = np.asarray(strain, dtype=float)
    temp = np.asarray(temp, dtype=float)
    
    if not np.all(np.isfinite(p)):
        return np.zeros_like(strain)
    
    T = temp
    abs_strain = np.abs(strain)
    
    # Temperature-dependent material parameters
    E = p[0] + p[1] * T       # Elastic modulus
    K = p[2] + p[3] * T       # Hardening coefficient
    n = p[4]                  # Hardening exponent
    C = p[5] + p[6] * T       # Offset
    
    # Clip n to valid range for power law
    n_safe = np.clip(n, 0.01, 100.0)
    
    # Compute |strain|^n * sign(strain) using smooth formulation
    # For numerical stability, use: strain * (strain^2 + eps)^((n-1)/2)
    # This equals sign(strain) * |strain|^n when eps << strain^2
    eps = 1e-10
    s2 = strain * strain
    base = s2 + eps
    exponent = (n_safe - 1.0) / 2.0
    ro_term = strain * np.power(base, exponent)
    
    # Elastic + hardening + offset
    elastic = E * strain
    hardening = K * ro_term
    sigma = elastic + hardening + C
    
    # Ensure output is finite
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    
    return sigma
