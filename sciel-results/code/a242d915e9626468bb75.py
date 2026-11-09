"""Bootstrap portfolio: model-and-optimize for stress-strain of aluminium rod."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIMEOUT_LIMIT = 140.0

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature.
    
    Uses a multiplicative interaction model that decouples:
    - Linear elastic scaling in strain
    - Temperature-dependent stiffness modulation
    - Non-linear strain saturation via power law
    - Additional rational saturation in both strain and temperature
    
    Model:
    stress = (p0 + p1*strain) * (p2 + p3*temp) * (1 + p4*strain^p5) 
             / (1 + p6*strain^p7 + p8*temp^p9) + p10
    """
    if time.time() - _START_TIME > _TIMEOUT_LIMIT:
        return np.zeros_like(strain, dtype=np.float64)

    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Clip strain for power operations to avoid negative base with non-integer exponent
    # Use a smooth approximation: for power p, if p is near integer it's fine,
    # but for safety, we'll use a different approach.
    # Actually, let's reconsider: strain can be negative. 
    # A better structural change: use a different form that handles negative strain naturally.
    
    # Let me try a different multiplicative form that's safe for negative strain:
    # stress = (p0 + p1*strain + p2*strain^2) * (p3 + p4*temp) * (1 + p5*strain) / (1 + p6*strain^2 + p7*temp^2) + p8 + p9*strain
    
    # Actually, the directive specifically asks for the form with strain^p5 and strain^p7.
    # Let's implement it carefully with safe power handling.
    
    # For strain^p5 where p5 may not be integer, we need |strain|^p5 * sign(strain) if p5 is odd-ish
    # Or better: use a smooth function that's well-defined for all real strain.
    
    # Alternative: replace strain^p with a smooth approximation
    # Let's use: f(x, a) = x * (1 + a*x^2)^(1/2) style, but that changes the form.
    
    # Actually, let's just implement the requested form but handle the power safely:
    # For non-integer powers of negative numbers, np.power gives nan.
    # We'll use: sign(strain) * |strain|^p for the power terms, which is smooth for p>0.
    
    # Extract parameters
    a0, a1, a2, a3, a4, a5, a6, a7, a8, a9 = params
    
    # Safe power function: for x^p, use sign(x)*|x|^p to handle negative x
    def safe_power(x, p):
        return np.sign(x) * np.power(np.abs(x), p)
    
    # Compute each factor
    # Factor 1: linear in strain
    F1 = a0 + a1 * strain
    
    # Factor 2: linear in temperature
    F2 = a2 + a3 * temp
    
    # Factor 3: nonlinear strain saturation (1 + a4 * strain^a5)
    strain_pow = safe_power(strain, a5)
    F3 = 1.0 + a4 * strain_pow
    
    # Denominator: 1 + a6 * strain^a7 + a8 * temp^a9
    strain_pow2 = safe_power(strain, a7)
    temp_pow = safe_power(temp, a9)
    denom = 1.0 + a6 * strain_pow2 + a8 * temp_pow
    
    # Safe division
    denom_safe = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
    
    stress = F1 * F2 * F3 / denom_safe + a9
    
    # Ensure finite output
    stress = np.where(np.isfinite(stress), stress, 0.0)
    
    return stress
# EVOLVE-BLOCK-END
