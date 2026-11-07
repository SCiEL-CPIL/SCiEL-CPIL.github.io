import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid Voce + Ramberg-Osgood model for stress-strain of an aluminium rod.
    
    Combines:
    - Elastic response: E(T) * strain
    - Voce-type exponential saturation: Q(T) * (1 - exp(-|strain| / H(T)))
    - Ramberg-Osgood power-law hardening: K(T) * strain * (strain^2 + delta)^((n(T)-1)/2)
    - Temperature-dependent parameters for all material constants
    
    sigma = E(T)*strain + (sigma_y(T) + Q(T)*(1-exp(-|strain|/H(T)))) * smooth_sign(strain)
            + K(T) * strain * (strain^2 + delta)^((n(T)-1)/2)
    
    Where:
    sigma_y(T) = p0 + p1*T
    Q(T)       = p2 + p3*T
    H(T)       = p4 + p5*T
    E(T)       = p6 + p7*T
    K(T)       = p8 + p9*T
    n(T)       = 1.0 (fixed, to keep parameter count at 10)
    
    Actually, let me reconsider the parameter allocation:
    p0, p1: sigma_y(T) = p0 + p1*T
    p2, p3: Q(T) = p2 + p3*T
    p4, p5: H(T) = p4 + p5*T
    p6, p7: E(T) = p6 + p7*T
    p8:     K (hardening coefficient, constant)
    p9:     n (hardening exponent, constant)
    
    The smooth sign uses tanh for differentiability, helping BFGS convergence.
    Centering strain around its mean reduces correlation between terms.
    """
    p = params
    
    strain = np.asarray(strain, dtype=float)
    temp = np.asarray(temp, dtype=float)
    
    if not np.all(np.isfinite(p)):
        return np.zeros_like(strain)
    
    T = temp
    abs_strain = np.abs(strain)
    
    # Temperature-dependent material parameters
    sigma_y = p[0] + p[1] * T       # Yield stress
    Q = p[2] + p[3] * T             # Saturation stress
    H = p[4] + p[5] * T             # Hardening modulus
    E = p[6] + p[7] * T             # Elastic modulus
    K = p[8]                        # Ramberg-Osgood hardening coefficient
    n = p[9]                        # Ramberg-Osgood hardening exponent
    
    # Clip H to avoid numerical issues in exp
    H_safe = np.clip(H, 1e-3, 1e4)
    
    # Exponential hardening term with clipped exponent
    exponent = -abs_strain / H_safe
    exponent = np.clip(exponent, -40.0, 0.0)
    exp_term = np.exp(exponent)
    voce = Q * (1.0 - exp_term)
    
    # Smooth sign via tanh: width scales with H for smooth elastic-plastic transition
    w = np.maximum(H_safe * 0.05, 1e-4)
    smooth_sign = np.tanh(strain / w)
    
    # Elastic part
    elastic = E * strain
    
    # Plastic part: yield + Voce hardening, gated by smooth sign
    plastic_voce = (sigma_y + voce) * smooth_sign
    
    # Ramberg-Osgood power-law hardening term
    # strain * (strain^2 + delta)^((n-1)/2) approximates |strain|^n * sign(strain)
    # For n=1, this is just strain (linear). For n>1, it's superlinear.
    n_safe = np.clip(n, 0.1, 50.0)
    delta = 1e-6
    s2 = strain * strain
    base = s2 + delta
    ro_exponent = (n_safe - 1.0) / 2.0
    ro_term = strain * np.power(base, ro_exponent)
    plastic_ro = K * ro_term
    
    sigma = elastic + plastic_voce + plastic_ro
    
    # Ensure output is finite
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    
    return sigma
