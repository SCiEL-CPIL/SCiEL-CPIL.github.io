import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model using a 5th order polynomial in strain,
    with coefficients linearly dependent on temperature.
    
    sigma = (p0 + p1*T)*u + (p2 + p3*T)*u^2 + (p3 + p5*T)*u^3 + (p4 + p7*T)*u^4 + (p8 + p9*T)*u^5
    Note: I need 10 params.
    Let:
    a1 = p0 + p1*T
    a2 = p2 + p3*T
    a3 = p4 + p5*T
    a4 = p6 + p7*T
    a5 = p8 + p9*T
    
    sigma = a1*u + a2*u^2 + a3*u^3 + a4*u^4 + a5*u^5
    """
    u = strain
    T = temp
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    a1 = p0 + p1 * T
    a2 = p2 + p3 * T
    a3 = p4 + p5 * T
    a4 = p6 + p7 * T
    a5 = p8 + p9 * T
    
    # Compute powers
    u2 = u * u
    u3 = u2 * u
    u4 = u2 * u2
    u5 = u4 * u
    
    sigma = a1 * u + a2 * u2 + a3 * u3 + a4 * u4 + a5 * u5
    
    return sigma
