"""Stress-strain model for aluminium rod with temperature dependence."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a Ramberg-Osgood-inspired rational model with temperature-dependent
    elastic modulus and yield strain, plus polynomial correction terms.
    
    Model:
        E(T)    = p0 + p1*T + p2*T^2
        eps_y(T) = p3 + p4*T
        ratio    = strain / eps_y(T)
        stress   = E(T) * strain / (1 + |ratio|^n) 
                   + p6*strain^2 + p7*T*strain + p8*T^2 + p9
    
    where n = p5 is the hardening exponent.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Temperature-dependent elastic modulus
    E = p0 + p1 * temp + p2 * temp * temp
    
    # Temperature-dependent yield strain (ensure positive)
    eps_y = p3 + p4 * temp
    # Guard against zero/negative eps_y
    eps_y_safe = np.where(np.abs(eps_y) < 1e-12, np.sign(eps_y) * 1e-12 + 1e-12, eps_y)
    eps_y_safe = np.where(eps_y_safe <= 0, 1e-12, eps_y_safe)
    
    # Ratio of strain to yield strain
    ratio = strain / eps_y_safe
    
    # Hardening exponent (ensure positive)
    n = np.maximum(p5, 0.1)
    
    # Main rational term: elastic-plastic transition
    # Using |ratio|^n to handle negative strains symmetrically
    denom = 1.0 + np.abs(ratio) ** n
    main_term = E * strain / denom
    
    # Polynomial correction terms
    correction = (p6 * strain * strain + 
                  p7 * temp * strain + 
                  p8 * temp * temp + 
                  p9)
    
    stress = main_term + correction
    
    return stress
# EVOLVE-BLOCK-END
