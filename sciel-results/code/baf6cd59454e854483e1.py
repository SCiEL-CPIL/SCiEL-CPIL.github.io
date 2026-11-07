import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Ramberg-Osgood type model for stress-strain of an aluminium rod.
    
    sigma = E(T)*eps + K(T)*|eps|^n * sign(eps) + C(T)*eps^3
    
    Where:
    E(T) = p0 + p1*T       (Elastic modulus)
    K(T) = p2 + p3*T       (Plastic modulus)
    n(T) = p4 + p5*T       (Hardenin exponent, clipped to [0.5, 20])
    C(T) = p6 + p7*T       (Cubic correction)
    p8, p9: additional terms for flexibility
    
    This power-law form captures the nonlinear hardening of aluminium
    more naturally than exponential saturation for many regimes.
    """
    p = params
    
    strain = np.asarray(strain, dtype=float)
    temp = np.asarray(temp, dtype=float)
    
    if not np.all(np.isfinite(p)):
        return np.zeros_like(strain)
    
    T = temp
    abs_strain = np.abs(strain)
    
    # Temperature-dependent parameters
    E = p[0] + p[1] * T
    K = p[2] + p[3] * T
    n = p[4] + p[5] * T
    C = p[6] + p[7] * T
    
    # Clip n to avoid numerical issues
    n_safe = np.clip(n, 0.5, 20.0)
    
    # Clip K to avoid overflow
    K_safe = np.clip(K, -1e6, 1e6)
    
    # Smooth sign via tanh for differentiability
    w = np.maximum(1e-4, np.abs(strain) * 1e-3)
    smooth_sign = np.tanh(strain / w)
    
    # Elastic part
    elastic = E * strain
    
    # Plastic part: power-law hardening
    # Use log to compute |eps|^n safely
    log_abs_strain = np.log(np.maximum(abs_strain, 1e-12))
    power_term = np.exp(n_safe * log_abs_strain)
    plastic = K_safe * power_term * smooth_sign
    
    # Cubic correction
    cubic = C * strain**3
    
    # Additional correction terms
    correction = p[8] * strain + p[9] * strain * T
    
    sigma = elastic + plastic + cubic + correction
    
    # Guard against non-finite values
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    
    return sigma
