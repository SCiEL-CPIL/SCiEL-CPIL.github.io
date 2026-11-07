"""Stress-strain of an aluminium rod: Separable polynomial model."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    stress = P(strain) + Q(temp) + R(strain, temp)
    
    P(e) = a0 + a1*e + a2*e^2 + a3*e^3 + a4*e^4
    Q(T) = b0*T + b1*T^2
    R(e, T) = c0*(e*T) + c1*(e^2*T) + c2*(e^3*T)
    
    Total 10 parameters: a0..a4, b0, b1, c0, c1, c2.
    This model is smooth, closed-form, and BFGS-friendly.
    """
    a0, a1, a2, a3, a4, b0, b1, c0, c1, c2 = params
    
    e = strain
    T = temp
    
    # Polynomial in strain
    p_e = a0 + a1 * e + a2 * e**2 + a3 * e**3 + a4 * e**4
    
    # Polynomial in temperature
    p_T = b0 * T + b1 * T**2
    
    # Interaction terms
    inter = c0 * (e * T) + c1 * (e**2 * T) + c2 * (e**3 * T)
    
    stress = p_e + p_T + inter
    return stress
# EVOLVE-BLOCK-END
