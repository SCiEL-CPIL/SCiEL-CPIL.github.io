import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Model: Multiplicative coupling of temperature-dependent elastic modulus
    with a polynomial strain base, plus sinh-based hardening for smooth saturation.
    
    sigma = E(T) * eps * (1 + a1*eps + a2*eps^2) + A(T) * sinh(B(T)*eps) / (1 + C(T)*eps^2)
    
    The multiplicative structure allows direct physical interplay between
    temperature and strain. The sinh term provides smooth yield-to-hardening
    transition with natural saturation via the denominator normalization.
    """
    eps = strain
    T = temp
    
    # Temperature-dependent elastic modulus
    E_T = params[0] * (1.0 + params[1] * T)
    
    # Polynomial strain correction coefficients (fitted)
    a1 = params[2]
    a2 = params[3]
    
    # Hardening amplitude with temperature dependence
    A_T = params[4] * (1.0 + params[5] * T)
    
    # Hardening rate with temperature dependence
    B_T = params[6] * (1.0 + params[7] * T)
    
    # Saturation normalization coefficient
    C_T = params[8] * (1.0 + params[9] * T)
    
    # Elastic-plastic base: E(T) * eps * polynomial correction
    base = E_T * eps * (1.0 + a1 * eps + a2 * eps**2)
    
    # Hardening term: sinh provides smooth saturation
    # Numerator: sinh(B*eps) grows exponentially but is bounded by denominator
    # Denominator: 1 + C*eps^2 provides quadratic saturation
    hardening = A_T * np.sinh(B_T * eps) / (1.0 + C_T * eps**2)
    
    stress = base + hardening
    
    return stress
