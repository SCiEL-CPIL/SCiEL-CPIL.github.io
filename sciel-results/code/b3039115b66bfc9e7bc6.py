# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for aluminium rod using piecewise-linear (ramp) approximation.
    
    Uses a sum of clipped linear terms (ramp functions) to model:
    - Elastic regime (linear)
    - Yield point transition
    - Strain hardening regime
    
    Parameters:
        strain: array of strain values (negative to positive)
        temp: array of temperature values (normalized 0-1)
        params: array of 10 fitted constants
            p0: intercept
            p1: elastic slope (strain)
            p2: temp slope
            p3: yield strain breakpoint 1
            p4: yield strain breakpoint 2
            p5: post-yield slope 1
            p6: post-yield slope 2
            p7: temp interaction with strain
            p8: temp interaction with post-yield
            p9: offset for temperature effect
    """
    eps = strain
    T = temp
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Base elastic-linear term
    base = p0 + p1 * eps + p2 * T
    
    # Ramp function: max(0, x - a) creates a kink at a
    # First ramp: captures yield onset
    ramp1 = np.maximum(0.0, eps - p3)
    
    # Second ramp: captures hardening regime
    ramp2 = np.maximum(0.0, eps - p4)
    
    # Combine with slopes
    stress = base + p5 * ramp1 + p6 * ramp2
    
    # Temperature interactions
    stress += p7 * T * eps
    stress += p8 * T * ramp1
    stress += p9 * T * ramp2
    
    return stress
# EVOLVE-BLOCK-END
