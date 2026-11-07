import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a polynomial model with temperature-dependent coefficients.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    eps = strain
    T = temp
    
    # Coefficients for powers of strain, each linear in temperature
    c0 = p0 + p1 * T
    c1 = p2 + p3 * T
    c2 = p4 + p5 * T
    c3 = p6 + p7 * T
    c4 = p8 + p9 * T
    
    # Polynomial expansion
    result = c0 + c1 * eps + c2 * eps**2 + c3 * eps**3 + c4 * eps**4
    
    return result
