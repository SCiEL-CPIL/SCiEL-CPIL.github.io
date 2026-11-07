"""Stress-strain model for aluminium rod with temperature dependence."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Combines a smooth elastic-plastic hinge (from inspiration 1) with
    temperature-dependent parameters and polynomial corrections.
    
    Model:
        E(T) = p0 + p1*T          (elastic modulus)
        Y(T) = p2 + p3*T          (yield stress plateau)
        k(T) = p4 + p5*T          (transition sharpness, clamped positive)
        
        core = E(T) * strain * exp(-k(T)*strain) + Y(T) * (1 - exp(-k(T)*strain))
        
        For small strain: core ≈ (E + Y*k) * strain  (linear elastic)
        For large strain: core → Y(T)                 (yield plateau)
        
        Plus polynomial corrections:
        p6*strain + p7*temp + p8*strain*temp + p9*temp^2
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Temperature-dependent parameters
    E = p0 + p1 * temp
    Y = p2 + p3 * temp
    k = p4 + p5 * temp
    
    # Clamp k to a positive range for numerical stability
    k_safe = np.clip(k, 0.1, 100.0)
    
    # Core elastic-plastic transition
    # For negative strain, the exponential behavior is asymmetric,
    # which is physically reasonable for tension/compression
    exp_term = np.exp(-k_safe * strain)
    core = E * strain * exp_term + Y * (1.0 - exp_term)
    
    # Polynomial correction terms for fine-tuning
    correction = (p6 * strain + 
                  p7 * temp + 
                  p8 * strain * temp + 
                  p9 * temp * temp)
    
    stress = core + correction
    
    return stress
# EVOLVE-BLOCK-END
