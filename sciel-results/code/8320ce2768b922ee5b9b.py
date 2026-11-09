"""Bootstrap portfolio: model-and-optimize for stress-strain of aluminium rod."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Record the start time to implement a deadline guard.
# The evaluator runs for up to 150s. We will fail gracefully after 140s.
_START_TIME = time.time()
_TIMEOUT_LIMIT = 140.0

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature.
    
    Uses a hybrid model combining a saturating rational term (elastic-plastic 
    transition) and a tanh term (strain hardening/saturation).
    
    This structure follows the high-performing Inspiration 1, which uses:
    - Quadratic temperature dependence for Elastic Modulus E(T).
    - Constant Hardening parameter H for the rational term denominator.
    - Linear temperature dependence for Hardening Stress S(T), Rate K(T), and Offset B(T).
    
    Model:
    stress = E(T)*strain / (1 + H*strain) + S(T)*tanh(K(T)*strain) + B(T)
    
    Where:
    E(T) = p0 + p1*temp + p2*temp**2
    H    = p3
    S(T) = p4 + p5*temp
    K(T) = p6 + p7*temp
    B(T) = p8 + p9*temp
    """
    # Deadline guard: if we are running out of time, return a safe neutral value
    if time.time() - _START_TIME > _TIMEOUT_LIMIT:
        return np.zeros_like(strain, dtype=np.float64)

    # Ensure inputs are 1D float64 arrays
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute temperature-dependent coefficients
    # Quadratic dependence for Elasticity to better capture softening
    E_T = p0 + p1 * temp + p2 * temp**2
    
    # Constant Hardening modulus (saturation strain parameter)
    # We add a small epsilon to prevent H from being exactly 0 if it causes issues, 
    # but typically BFGS will find a non-zero value. 
    # However, if H is very small, the term becomes linear, which is fine.
    H = p3
    
    # Linear dependence for Saturation stress and Rate
    S_T = p4 + p5 * temp
    K_T = p6 + p7 * temp
    
    # Linear dependence for Offset
    B_T = p8 + p9 * temp
    
    # Term 1: Saturating rational function
    # For small strain, approx E_T * strain (Elastic)
    # For large strain, approx E_T / H (Saturation level)
    # Safe division
    den = 1.0 + H * strain
    
    # Avoid division by zero or near-zero denominators which cause numerical instability
    # and gradients that are too large for BFGS.
    # If |den| < eps, we replace it with eps * sign(den) to maintain continuity 
    # in direction but cap the magnitude.
    eps = 1e-6
    den_safe = np.where(np.abs(den) < eps, np.sign(den + 1e-300) * eps, den)
    
    term1 = (E_T * strain) / den_safe
    
    # Term 2: Tanh saturation for hardening
    # tanh is bounded [-1, 1], smooth, captures the "knee" and plateau
    # Clip the argument to tanh to avoid any potential precision issues with 
    # extremely large values, though tanh handles this well in float64.
    # tanh(>20) is effectively 1.
    tanh_arg = K_T * strain
    # Optional: np.clip(tanh_arg, -50, 50) for safety, but usually not needed.
    term2 = S_T * np.tanh(tanh_arg)
    
    # Term 3: Offset
    term3 = B_T
    
    stress = term1 + term2 + term3
    
    # Ensure output is finite
    stress = np.where(np.isfinite(stress), stress, 0.0)
    
    return stress
# EVOLVE-BLOCK-END
