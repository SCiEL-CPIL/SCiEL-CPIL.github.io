"""Rational function model for aluminium rod stress-strain."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress = (E(T) * strain + H(T) * strain^2) / (1 + A(T) * strain + B(T) * strain^2)
    
    This rational form can mimic the elastic-plastic transition:
    - For small strain, it behaves like a linear function (elastic).
    - For large strain, it saturates or follows a different trend depending on coefficients.
    - The denominator allows for complex curvature changes that additive polynomials/exponentials 
      struggle to fit efficiently with limited parameters.
    
    Parameters:
    p0, p1: E(T) = p0 + p1*temp (Elastic modulus)
    p2, p3: H(T) = p2 + p3*temp (Hardening coefficient)
    p4, p5: A(T) = p4 + p5*temp (Linear denominator coeff)
    p6, p7: B(T) = p6 + p7*temp (Quadratic denominator coeff)
    p8, p9: Offset and scaling? Let's use p8 as a constant offset and p9 as a scaling factor 
            to ensure we have 10 params.
            
    Actually, let's stick to a pure rational function for the shape, and use the last two 
    params for a temperature-dependent shift and scale if needed, or just leave them 
    in the coefficients.
    
    Let's define:
    Num = (p0 + p1*temp) * strain + (p2 + p3*temp) * strain**2
    Den = 1.0 + (p4 + p5*temp) * strain + (p6 + p7*temp) * strain**2
    
    We have 8 params. We need 2 more.
    Let's add a linear term in the numerator that is not just E*strain, but allows 
    for a yield offset? No, stress should be 0 at strain 0.
    
    Let's use:
    Num = (p0 + p1*temp) * strain + (p2 + p3*temp) * strain**2 + (p8 + p9*temp) * strain**3
    Den = 1.0 + (p4 + p5*temp) * strain + (p6 + p7*temp) * strain**2
    
    This gives a cubic numerator and quadratic denominator. This is a very flexible 
    rational function that can fit S-shaped or saturating curves well.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Temperature-dependent coefficients
    c_e = p0 + p1 * temp
    c_h = p2 + p3 * temp
    c_a = p4 + p5 * temp
    c_b = p6 + p7 * temp
    c_c = p8 + p9 * temp
    
    # Numerator: cubic in strain
    num = c_e * strain + c_h * strain**2 + c_c * strain**3
    
    # Denominator: quadratic in strain
    den = 1.0 + c_a * strain + c_b * strain**2
    
    # Prevent division by zero or near-zero
    # In the given range strain is [-0.0019, 0.96]. 
    # If den becomes 0, we have issues. We can clip den to a small epsilon.
    # However, BFGS might explore regions where den is close to 0.
    # A safe approach is to add a small constant to the denominator to ensure it's non-zero,
    # or clip the result.
    # Let's clip the denominator to avoid singularities.
    # Since strain is mostly positive and small, and we start with 1.0, den is likely positive.
    # But to be safe:
    den_safe = np.where(np.abs(den) < 1e-8, 1e-8 * np.sign(den), den)
    # If den is 0, sign is 0, so 1e-8 * 1 (or -1). Let's just use a small positive number 
    # if abs(den) is small, assuming the "1.0" base keeps it positive mostly.
    # Actually, simpler:
    den_safe = np.where(np.abs(den) < 1e-8, 1e-8, den)
    
    return num / den_safe
# EVOLVE-BLOCK-END
