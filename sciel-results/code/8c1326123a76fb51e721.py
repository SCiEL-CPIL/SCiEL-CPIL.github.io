import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a hyperbolic model for the elastic-plastic transition
    combined with polynomial terms for hardening and temperature effects.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure inputs are arrays for broadcasting
    strain = np.asarray(strain)
    temp = np.asarray(temp)
    
    # Tanh-based smooth saturation: captures yield point and strain hardening
    # p0: amplitude of plastic response
    # p1: temperature coefficient on plastic amplitude
    # p2: steepness of yield transition
    # p3: temperature-dependent yield strain shift
    yield_arg = p2 * (strain - p3 * temp)
    tanh_term = p0 * np.tanh(yield_arg) * (1.0 + p1 * temp)
    
    # Polynomial terms: capture linear elasticity, plastic hardening, 
    # temperature dependence, and interactions.
    poly_term = (
        p4 * strain + 
        p5 * temp + 
        p6 * temp**2 + 
        p7 * strain * temp + 
        p8 + 
        p9 * strain**2
    )
    
    return tanh_term + poly_term
