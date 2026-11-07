"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature."""
    # params: length 10
    # c0, c1, c2, c3, c4, c5, c6 : numerator coefficients
    # d1, d2, d3 : denominator coefficients (constant term is 1)
    
    c0, c1, c2, c3, c4, c5, c6, d1, d2, d3 = params
    
    # Compute numerator
    # P(e, T) = c0 + c1*e + c2*e^2 + c3*e^3 + c4*T + c5*e*T + c6*e^2*T
    num = c0 + c1 * strain + c2 * strain**2 + c3 * strain**3 + c4 * temp + c5 * strain * temp + c6 * strain**2 * temp
    
    # Compute denominator
    # Q(e, T) = 1 + d1*e + d2*e^2 + d3*T
    den = 1.0 + d1 * strain + d2 * strain**2 + d3 * temp
    
    # Avoid division by zero
    # If den is close to zero, the model is ill-conditioned.
    # We can clip the denominator to a small value.
    # However, clipping can cause issues with gradients.
    # Given the training range, if the fit is good, den should be safe.
    # To be safe, we add a small constant to the denominator? No, that changes the model.
    # Let's just do the division. If it fails, the score will be bad.
    
    stress = num / den
    
    return stress
# EVOLVE-BLOCK-END
