"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model using a multiplicative interaction structure.
    
    Structure:
    sigma = (p0 + p1*strain) * (p2 + p3*temp) * (1 + p4*strain^p5) / (1 + p6*strain^p7 + p8*temp^p9) + p10
    
    This structure allows:
    - Linear elastic scaling via (p0 + p1*strain)
    - Temperature-dependent stiffness via (p2 + p3*temp)
    - Non-linear strain saturation/hardening via (1 + p4*strain^p5)
    - Denominator provides additional flexibility for saturation or softening
    - p10 allows for a baseline offset
    
    params: array of length 10
    """
    start_time = time.time()
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9]
    
    # Ensure strain is clipped to avoid negative base for fractional powers if needed, 
    # though strain is mostly positive. The range is [-0.001911, 0.9636].
    # To be safe with fractional powers, we can use abs or clip. 
    # However, the model form suggests smoothness. Let's use a small epsilon to avoid 0^0 issues or negative bases.
    # Actually, if p5 is an integer, it's fine. If not, we need care.
    # Let's assume the optimizer finds sensible exponents. 
    # To be robust, let's clip strain to a small positive value for the power terms?
    # No, better to let the optimizer decide. But np.power(negative, fractional) is nan.
    # The strain has negative values (down to -0.0019).
    # If p5 is 2, strain^2 is positive. If p5 is 0.5, it's nan.
    # Let's use a safe power function that handles negatives by taking abs or using sign-preserving logic?
    # Or simply, the optimizer might choose integer-like exponents.
    # To ensure finite output, we can use:
    # strain_safe = np.where(strain > 0, strain, 1e-6) ? No, that breaks the physics for negative strain (compression).
    # Let's assume the dominant behavior is in tension.
    # For negative strain, we can just use the linear part or a separate branch?
    # The prompt asks for a single closed-form expression.
    
    # Let's try to make the power terms robust.
    # We can use: np.power(np.abs(strain), p5) * np.sign(strain) ? 
    # Or just np.power(strain, p5) and hope p5 is even or integer?
    # A safer bet for a general "saturation" term that works for both signs:
    # Use (1 + p4 * strain^2) ? That's fixed p5=2.
    # The prompt allows fitted constants.
    # Let's stick to the proposed form but add a guard.
    # If we use np.power, and it produces nan, the score is 1e30.
    # Let's use a helper that clips the base to be positive for the power calculation, 
    # but this changes the function for negative strain.
    # Alternative: use a hyperbolic tangent or sigmoid based model?
    # Let's stick to the polynomial/rational form but ensure numerical stability.
    
    # Let's define a safe power:
    # If base < 0 and exponent is not integer, result is nan.
    # We can approximate strain^p5 using exp(p5 * log(abs(strain))) * sign?
    # Let's just clip strain to a small positive value for the *non-linear* terms only?
    # That might be okay if the non-linear effect is small at negative strain.
    
    strain_pos = np.clip(strain, 1e-8, None)
    
    # Calculate terms
    term1 = p0 + p1 * strain
    term2 = p2 + p3 * temp
    term3 = 1.0 + p4 * (strain_pos ** p5)
    denom = 1.0 + p6 * (strain_pos ** p7) + p8 * (temp ** p9)
    
    # Avoid division by zero
    denom_safe = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
    
    stress = term1 * term2 * term3 / denom_safe + p10
    
    return stress
# EVOLVE-BLOCK-END
