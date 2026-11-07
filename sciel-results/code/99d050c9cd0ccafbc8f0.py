import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for aluminium rod.
    
    Structural change: Hybrid of rational hardening (inspiration 3) and 
    temperature-modulated polynomial corrections (current program).
    
    Model:
      E(T) = p0 + p1 * temp
      Core elastic: E(T) * strain
      Hardening: (p2 + p3*T)*strain^2 + (p4 + p5*T)*strain^3
                 divided by (1 + (p6 + p7*T)*strain + (p8 + p9*T)*strain^2)
      
    This separates the linear elastic response from the non-linear plastic 
    hardening, which is captured by a rational function to allow for 
    saturation or complex curvature, while maintaining smoothness.
    """
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Temperature-dependent elastic modulus
    E = p0 + p1 * temp
    
    # Linear elastic component
    sigma_elastic = E * strain
    
    # Hardening numerator coefficients (temperature dependent)
    H2 = p2 + p3 * temp
    H3 = p4 + p5 * temp
    
    # Hardening denominator coefficients (temperature dependent)
    K1 = p6 + p7 * temp
    K2 = p8 + p9 * temp
    
    eps2 = strain * strain
    eps3 = eps2 * strain
    
    numerator = H2 * eps2 + H3 * eps3
    denominator = 1.0 + K1 * strain + K2 * eps2
    
    # Safety guard against division by zero
    # Ensure denominator doesn't get too close to zero to avoid instability in BFGS
    # If denominator is very small, clamp it to a small positive value
    # Note: For physical reasons, denominator should be positive in the training range.
    # If it goes negative, the sign flip might be physical (softening) or numerical artifact.
    # To be safe for optimization, we prevent exact zero.
    # Using np.where to handle potential zeros robustly.
    # However, simple addition of a tiny epsilon is often sufficient if the scale is 1.0.
    # Let's use a robust clamp: if |denom| < 1e-6, set to 1e-6 * sign(denom) or 1e-6.
    # Since we expect denom > 0, we can just clamp to 1e-6 if it drops below.
    # But what if it's legitimately negative? 
    # Given the form 1 + K*eps, and eps in [-0.002, 0.96], if K is large negative, it could cross 0.
    # A rational function that crosses zero in the denominator is singular.
    # The BFGS optimizer will likely avoid this region if it causes large errors.
    # To ensure finite outputs, we clamp.
    
    # A safer approach for rational functions in optimization is to add a small constant
    # to the denominator to shift the pole away from the domain of interest, or use
    # a smooth approximation like tanh. But simple clamping is standard.
    
    denom_safe = np.where(np.abs(denominator) < 1e-6, 1e-6 * np.where(denominator >= 0, 1.0, -1.0), denominator)
    
    sigma_hardening = numerator / denom_safe
    
    return sigma_elastic + sigma_hardening
