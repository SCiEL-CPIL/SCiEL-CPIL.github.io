import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a saturating exponential model with polynomial corrections.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    eps = strain
    T = temp
    
    # Temperature-dependent coefficients
    # Saturation Amplitude A(T)
    A = p0 + p1 * T
    
    # Saturation Rate B(T) - must be positive for stability
    B_raw = p2 + p3 * T
    B = np.where(B_raw > 1e-6, B_raw, 1e-6)
    
    # Linear Coefficient C(T)
    C = p4 + p5 * T
    
    # Quadratic Coefficient D(T)
    D = p6 + p7 * T
    
    # Cubic Coefficient E(T)
    E = p8 + p9 * T
    
    # Calculate terms
    # Term 1: Saturating exponential (models hardening/plateau)
    # For small eps, 1 - exp(-B*eps) ~ B*eps
    term_sat = A * (1.0 - np.exp(-B * eps))
    
    # Term 2: Linear elastic/plastic
    term_lin = C * eps
    
    # Term 3: Quadratic correction
    term_quad = D * eps**2
    
    # Term 4: Cubic correction
    term_cub = E * eps**3
    
    result = term_sat + term_lin + term_quad + term_cub
    
    return result
