"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a multiplicative/modulated structure with a rational hardening function
    and piecewise-linear "sawtooth" basis to model discrete yield points and
    work-hardening segments.
    
    Model:
    sigma = E(T)*eps + Y(T)*[H(eps) + S(eps)] + p9
    E(T) = p0 + p1*T + p2*T^2
    Y(T) = p3 + p4*T
    H(eps) = (p5*eps + p6*eps^2) / (1 + p7*eps + p8*eps^2)
    S(eps) = p5 * (seg1 + 0.5*seg2)  # piecewise-linear segments
    
    params (10 constants):
      p0, p1, p2: Elastic modulus coefficients
      p3, p4: Yield/Hardening scale coefficients
      p5: Hardening numerator linear coeff / segment amplitude
      p6: Hardening numerator quadratic coeff
      p7: Hardening denominator linear coeff
      p8: Hardening denominator quadratic coeff / segment period
      p9: Constant offset
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    eps = strain
    T = temp
    T2 = T * T
    eps2 = eps * eps
    
    # Temperature-dependent elastic modulus
    E_T = p0 + p1 * T + p2 * T2
    
    # Temperature-dependent hardening scale
    Y_T = p3 + p4 * T
    
    # Rational hardening function
    # Numerator: p5*eps + p6*eps^2
    # Denominator: 1 + p7*eps + p8*eps^2
    # Safety: ensure denominator is not near zero
    denom = 1.0 + p7 * eps + p8 * eps2
    safe_denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
    
    H_eps = (p5 * eps + p6 * eps2) / safe_denom
    
    # Piecewise-linear sawtooth basis for discrete yield points
    # Use a safe period from p8 (denominator quadratic coeff)
    period = np.where(np.abs(p8) > 1e-8, p8, 1.0)
    
    # Create hardening segments using max operations (differentiable a.e.)
    seg1 = np.maximum(eps, 0) - np.maximum(eps - period, 0)
    seg1 = seg1 / np.where(np.abs(period) > 1e-8, np.abs(period), 1.0)
    
    seg2 = np.maximum(eps - period, 0) - np.maximum(eps - 2*period, 0)
    seg2 = seg2 / np.where(np.abs(period) > 1e-8, np.abs(period), 1.0)
    
    # Combine segments with decreasing amplitude (work hardening saturation)
    S_eps = p5 * (seg1 + 0.5 * seg2)
    
    # Core model: elastic + temperature-scaled (rational + piecewise) + offset
    sigma = E_T * eps + Y_T * (H_eps + S_eps) + p9
    
    return sigma
# EVOLVE-BLOCK-END
