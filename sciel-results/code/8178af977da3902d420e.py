"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a piecewise-linear approximation via smooth ramp functions (softplus)
    to model elastic, yield, and strain hardening regimes.
    This provides a smooth, differentiable surface for BFGS optimization.
    """
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Temperature effect (centered)
    T = temp - 0.5
    
    # Smooth ramp function parameters
    alpha = 50.0
    
    def smooth_ramp(x, c):
        # Softplus approximation of max(0, x - c)
        # softplus(z) = log(1 + exp(z))
        # ramp(x, c) = softplus(alpha * (x - c)) / alpha
        z = alpha * (x - c)
        return np.log1p(np.exp(z)) / alpha

    # Basis functions
    # Elastic region
    b0 = strain
    
    # Plastic yield and hardening regions
    # Yield points estimated for aluminium
    b1 = smooth_ramp(strain, 0.001)
    b2 = smooth_ramp(strain, 0.002)
    b3 = smooth_ramp(strain, 0.005)
    b4 = smooth_ramp(strain, 0.01)
    b5 = smooth_ramp(strain, 0.05)
    b6 = smooth_ramp(strain, 0.1)
    b7 = smooth_ramp(strain, 0.2)
    b8 = smooth_ramp(strain, 0.5)
    
    # Temperature term
    b9 = T
    
    # Intercept is handled by the linear combination? 
    # Wait, I have 10 params.
    # p0 * b0 + ... + p8 * b8 + p9 * 1?
    # Or p9 * b9?
    # In the previous model, p9 was intercept.
    # Let's use p9 as intercept.
    # Then where is temperature?
    # I have b0..b8 (9 terms).
    # If p9 is intercept, I have no temperature term.
    # The current model has temperature. Temperature is likely important.
    # Let's drop one ramp to include temperature?
    # Or make one of the params an interaction?
    # The directive says "Replace ... with piecewise-linear ...". It doesn't explicitly forbid temperature, but the model should be simple.
    # However, ignoring temperature might hurt if it's in the data.
    # Let's look at the inputs: strain, temp.
    # If I drop temperature, I might lose performance.
    # Let's use 8 ramps + 1 linear + 1 temp + 1 intercept? That's 11.
    # I have 10 params.
    # Let's use 7 ramps + linear + temp + intercept.
    # Ramps at: 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5.
    # Drop 0.002.
    
    b1 = smooth_ramp(strain, 0.001)
    b2 = smooth_ramp(strain, 0.005)
    b3 = smooth_ramp(strain, 0.01)
    b4 = smooth_ramp(strain, 0.05)
    b5 = smooth_ramp(strain, 0.1)
    b6 = smooth_ramp(strain, 0.2)
    b7 = smooth_ramp(strain, 0.5)
    
    stress = (
        p0 * b0 +
        p1 * b1 +
        p2 * b2 +
        p3 * b3 +
        p4 * b4 +
        p5 * b5 +
        p6 * b6 +
        p7 * b7 +
        p8 * T +
        p9
    )
    
    return stress
# EVOLVE-BLOCK-END
