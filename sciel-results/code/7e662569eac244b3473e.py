"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a rational power-law form to capture elastic-plastic behavior:
    - Initial linear elastic response
    - Strain hardening via a power-law term
    - Saturation/softening captured by the rational denominator
    - Temperature-dependent yield and hardening coefficients
    
    The form is:
    stress = (a0 + a1*T + a2*T^2) * strain 
           + (b0 + b1*T) * strain^p / (1 + (strain/c)^q)
    """
    s = np.asarray(strain, dtype=np.float64)
    t = np.asarray(temp, dtype=np.float64)
    
    # Ensure strain is non-negative for power operations (plastic strain)
    s_pos = np.maximum(s, 0.0)
    
    # Extract parameters
    # Elastic modulus (with temperature dependence)
    E0 = params[0]
    E1 = params[1]
    # Yield-related terms
    b0 = params[2]
    b1 = params[3]
    # Power-law exponent for hardening
    p = params[4]
    # Saturation strain scale
    c = params[5]
    # Denominator exponent
    q = params[6]
    # Additional temperature correction for hardening
    T_corr = params[7]
    # Offset
    offset = params[8]
    # Additional strain-dependent term coefficient
    extra = params[9]
    
    # Temperature-dependent elastic modulus
    E = E0 + E1 * t
    
    # Temperature-dependent hardening coefficient
    b = b0 + b1 * t + T_corr * t * t
    
    # Ensure c is positive for the rational form
    c_safe = np.maximum(c, 1e-6)
    
    # Ensure exponents are reasonable (avoid 0 or negative which could cause issues)
    p_safe = np.maximum(p, 0.01)
    q_safe = np.maximum(q, 0.01)
    
    # Plastic strain term with rational saturation
    # strain^p / (1 + (strain/c)^q)
    s_pow = np.power(s_pos, p_safe)
    ratio = np.power(s_pos / c_safe, q_safe)
    plastic_term = b * s_pow / (1.0 + ratio)
    
    # Additional higher-order strain term for better fit
    s2 = s_pos * s_pos
    extra_term = extra * s2 / (1.0 + s_pos)
    
    # Combine
    stress = E * s + plastic_term + extra_term + offset
    
    return stress
# EVOLVE-BLOCK-END
