import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for aluminium rod.
    Combines a smooth elastic-plastic transition (rational saturation)
    with temperature-dependent coefficients and a smooth hardening term.
    
    Structure:
      E(T) = p0 + p1 * temp
      eps_y(T) = p2 + p3 * temp
      sigma = E(T) * strain / (1 + (strain / eps_y(T))^2 * p4)
              + (p5 + p6 * temp) * (strain - eps_y(T))_+^p7
              + p8 * temp + p9 * temp^2
    
    The rational form E*strain/(1+(strain/eps_y)^2*p4) provides a smooth
    transition from linear elastic to a plateau, avoiding the kink of a
    hard hinge. The hardening term uses a smooth positive part.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Temperature-dependent elastic modulus
    E = p0 + p1 * temp
    
    # Temperature-dependent yield strain
    eps_y = p2 + p3 * temp
    
    # Safety: avoid division by zero in eps_y
    eps_y_safe = np.where(np.abs(eps_y) < 1e-6, 1e-6, eps_y)
    
    # Rational saturation term: smooth elastic-plastic transition
    # At small strain: ~ E * strain
    # At large strain: ~ E * eps_y^2 / (p4 * strain) -> decays, but we add hardening
    # Actually, this form gives a peak at strain = eps_y/sqrt(p4)
    # Let's use a different smooth form that plateaus:
    # sigma_elastic_plastic = E * strain / (1 + p4 * (strain / eps_y)^2)
    # This is smooth and differentiable everywhere.
    
    ratio = strain / eps_y_safe
    denom = 1.0 + p4 * ratio * ratio
    # Safety for denominator
    denom_safe = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
    
    elastic_plastic = E * strain / denom_safe
    
    # Smooth plastic hardening
    # Use a smooth approximation of max(strain - eps_y, 0)
    # s = strain - eps_y
    # smooth_pos = 0.5 * (s + sqrt(s^2 + eps^2))
    # But we want it to start at 0 when s=0 and grow.
    # Let's use: hardening = K(T) * (strain - eps_y) * smooth_step(strain - eps_y)
    # Or simpler: power law on the positive part, smoothed.
    
    s = strain - eps_y
    # Smooth positive part: 0.5*(s + sqrt(s^2 + 1e-6))
    s_pos_smooth = 0.5 * (s + np.sqrt(s * s + 1e-6))
    
    # Hardening coefficient
    K = p5 + p6 * temp
    
    # Hardening term
    # To avoid issues with negative base for fractional power, we use s_pos_smooth
    # which is always positive.
    hardening = K * np.power(s_pos_smooth, p7)
    
    # Temperature corrections
    temp_corr = p8 * temp + p9 * temp * temp
    
    return elastic_plastic + hardening + temp_corr
