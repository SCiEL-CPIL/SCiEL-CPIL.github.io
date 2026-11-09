"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a combined elastic-plastic model with temperature-dependent
    elastic modulus, yield strength, and saturation rate.
    
    Model: 
    sigma = E(T)*eps + Y(T)*tanh(K(T)*eps) + p7*eps^2 + p8*eps*T + p9
    E(T) = p0 + p1*T + p2*T^2
    Y(T) = p3 + p4*T
    K(T) = p5 + p6*T
    
    params (10 constants):
      p0, p1, p2: Elastic modulus coefficients
      p3, p4: Yield/Hardening scale coefficients
      p5, p6: Saturation rate coefficients
      p7: Quadratic strain correction
      p8: Strain-Temperature interaction
      p9: Constant offset
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    eps = strain
    T = temp
    
    # Temperature-dependent coefficients
    E_T = p0 + p1 * T + p2 * T * T
    Y_T = p3 + p4 * T
    K_T = p5 + p6 * T
    
    # Ensure K_T is not zero to avoid division issues if we used 1/K, 
    # but tanh is safe. However, if K_T is very large, tanh saturates quickly.
    # If K_T is negative, tanh is negative, which might be wrong for hardening.
    # We assume params will be fitted to positive values if needed, 
    # but to be safe, we can use abs or just let the optimizer handle it.
    # Given the task, hardening is positive, so K_T should be positive.
    
    # Core model
    elastic_term = E_T * eps
    plastic_term = Y_T * np.tanh(K_T * eps)
    
    # Correction terms
    quad_term = p7 * eps * eps
    inter_term = p8 * eps * T
    
    sigma = elastic_term + plastic_term + quad_term + inter_term + p9
    
    return sigma
# EVOLVE-BLOCK-END
