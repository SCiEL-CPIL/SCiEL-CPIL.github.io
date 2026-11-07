"""Stress-strain model for aluminium rod with temperature dependence."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a piecewise model with smooth exponential hinge to separate
    elastic and plastic regimes.
    
    Model:
        E(T)   = p0 + p1*T       (elastic modulus)
        s_y(T) = p2 + p3*T       (yield strain)
        Y(T)   = p4 + p5*T       (yield stress level)
        H(T)   = p6 + p7*T       (plastic modulus)
        n      = p8              (hardening exponent, clamped to [0.5, 3.0])
        k      = p9              (transition sharpness, clamped to [1, 100])
        
        elastic = E(T) * strain
        excess = max(strain - s_y, 0)
        smooth_excess = excess * (1 - exp(-k * excess / max(|s_y|, 1e-10)))
        plastic = Y(T) + H(T) * smooth_excess^n
        
        stress = elastic * (1 - step) + plastic * step
        where step = 1 - exp(-k * max(strain - s_y, 0) / max(|s_y|, 1e-10))
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Temperature-dependent parameters
    E = p0 + p1 * temp
    s_y = p2 + p3 * temp
    Y = p4 + p5 * temp
    H = p6 + p7 * temp
    n = np.clip(p8, 0.5, 3.0)
    k = np.clip(p9, 1.0, 100.0)
    
    # Scale for normalization
    scale = np.maximum(np.abs(s_y), 1e-10)
    
    # Excess strain beyond yield
    excess = np.maximum(strain - s_y, 0.0)
    
    # Smooth transition using exponential
    x = excess / scale
    # Clamp to avoid overflow in exp
    x_clamped = np.clip(x, 0.0, 50.0)
    step = 1.0 - np.exp(-k * x_clamped)
    
    # Elastic contribution
    elastic = E * strain
    
    # Smooth plastic excess
    excess_safe = np.maximum(excess, 1e-15)
    # Use the same smooth factor for the hardening term
    smooth_excess = excess_safe * (1.0 - np.exp(-k * np.clip(excess_safe / scale, 0.0, 50.0)))
    
    # Plastic contribution with power-law hardening
    hardening = H * np.power(smooth_excess, n)
    plastic = Y + hardening
    
    # Combine elastic and plastic with smooth transition
    stress = elastic * (1.0 - step) + plastic * step
    
    return stress
# EVOLVE-BLOCK-END
