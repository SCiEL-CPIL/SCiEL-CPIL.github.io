import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a tanh-based model for the elastic-plastic transition
    combined with polynomial terms for hardening and temperature effects.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure inputs are arrays for broadcasting
    strain = np.asarray(strain)
    temp = np.asarray(temp)
    
    # Tanh term: captures the smooth yield transition and saturation
    # p0: scale, p1: stiffness/transition sharpness, p2: temperature effect on yield
    hyper_term = p0 * np.tanh(p1 * strain) * (1.0 + p2 * temp)
    
    # Polynomial terms: capture linear elasticity, plastic hardening, 
    # temperature dependence, and interactions.
    # p3: linear strain, p4: linear temp, p5: quad temp, p6: strain-temp, p7: const
    # p8: quad strain, p9: cubic strain
    poly_term = (
        p3 * strain + 
        p4 * temp + 
        p5 * temp**2 + 
        p6 * strain * temp + 
        p7 + 
        p8 * strain**2 +
        p9 * strain**3
    )
    
    return hyper_term + poly_term
