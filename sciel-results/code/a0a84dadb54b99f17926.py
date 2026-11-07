import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Voce-type exponential hardening model for stress-strain of an aluminium rod.
    
    sigma = sigma_y(T) + Q(T) * (1 - exp(-strain / H(T)))
    
    Where:
    sigma_y(T) = p0 + p1*T          (Yield stress)
    Q(T)       = p2 + p3*T          (Saturation stress range)
    H(T)       = p4 + p5*T          (Hardening modulus / strain scale)
    
    This captures the asymptotic saturation of stress at high strains,
    a key feature of aluminium that power-law models fail to represent.
    
    Additional terms for better fit:
    p6: Linear elastic correction
    p7: Quadratic strain correction
    p8: Temperature-strain interaction
    p9: Constant offset
    """
    p = params
    
    # Safety check for non-finite parameters
    if not np.all(np.isfinite(p)):
        return np.zeros_like(strain)
    
    T = temp
    eps = strain
    
    # Temperature-dependent Voce parameters
    sigma_y = p[0] + p[1] * T
    Q = p[2] + p[3] * T
    H = p[4] + p[5] * T
    
    # Ensure H is positive to avoid division by zero and ensure exponential decay
    H_safe = np.where(np.abs(H) < 1e-6, 1e-6 * np.sign(H + 1e-12), H)
    H_safe = np.where(H_safe > 0, H_safe, 1e-6)
    
    # Core Voce term: Q * (1 - exp(-strain/H))
    # Clip argument to avoid overflow
    arg = eps / H_safe
    arg_clipped = np.clip(arg, -50, 50)
    voce_term = Q * (1.0 - np.exp(-arg_clipped))
    
    # Additional correction terms for better fit
    linear_correction = p[6] * eps
    quadratic_correction = p[7] * eps * eps
    temp_strain_interaction = p[8] * T * eps
    offset = p[9]
    
    sigma = sigma_y + voce_term + linear_correction + quadratic_correction + temp_strain_interaction + offset
    
    return sigma
