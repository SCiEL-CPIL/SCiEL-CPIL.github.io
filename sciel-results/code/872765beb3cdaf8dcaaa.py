"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a physically-motivated model with:
    - Temperature-dependent elastic modulus (quadratic)
    - Bilinear plastic hardening (tanh-based smooth approximation)
    - Strain-temperature interaction terms
    - Higher-order correction terms
    
    Model:
    sigma = E(T) * eps + Y(T) * tanh(K(T) * eps) + p7 * eps^2 + p8 * eps * T + p9 * T + p10_offset
    E(T) = p0 + p1 * T + p2 * T^2
    Y(T) = p3 + p4 * T
    K(T) = p5 + p6 * T
    
    Note: We use 10 parameters total (p0..p9 in the array).
    Let's restructure to use exactly 10:
    p0, p1, p2: E(T) coefficients (quadratic in T)
    p3, p4: Y(T) coefficients  
    p5, p6: K(T) coefficients
    p7: quadratic strain
    p8: eps*T interaction
    p9: T interaction (offset)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    eps = strain
    T = temp
    
    # Temperature-dependent coefficients
    E_T = p0 + p1 * T + p2 * T * T
    Y_T = p3 + p4 * T
    K_T = p5 + p6 * T
    
    # Core model: elastic + plastic hardening + corrections
    elastic_term = E_T * eps
    plastic_term = Y_T * np.tanh(K_T * eps)
    
    # Correction terms
    quad_term = p7 * eps * eps
    inter_term = p8 * eps * T
    temp_term = p9 * T
    
    sigma = elastic_term + plastic_term + quad_term + inter_term + temp_term
    
    return sigma
# EVOLVE-BLOCK-END
