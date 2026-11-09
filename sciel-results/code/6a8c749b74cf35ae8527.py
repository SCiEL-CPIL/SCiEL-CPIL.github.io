"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature."""
    # Model: Piecewise-linear approximation using smooth ramp functions.
    # Captures elastic region, yield, hardening, and temperature effects.
    # Parameters (10 total):
    # 0: a0 (intercept)
    # 1: a1 (elastic slope / linear strain term)
    # 2: a2 (linear temperature effect)
    # 3: b1 (yield strain breakpoint)
    # 4: a3 (slope change at yield)
    # 5: b2 (hardening breakpoint)
    # 6: a4 (slope change at hardening)
    # 7: b3 (late hardening breakpoint)
    # 8: a5 (slope change at late hardening)
    # 9: t1 (temperature breakpoint for non-linear temp effect)
    
    a0 = params[0]
    a1 = params[1]
    a2 = params[2]
    
    b1 = params[3]
    a3 = params[4]
    
    b2 = params[5]
    a4 = params[6]
    
    b3 = params[7]
    a5 = params[8]
    
    t1 = params[9]
    
    eps = 1e-8
    
    def smooth_ramp(x):
        # Smooth approximation of max(0, x) using softplus-like form
        return (x + np.sqrt(x**2 + eps)) / 2.0
    
    # Strain ramps for piecewise linear stress-strain curve
    r_s1 = smooth_ramp(strain - b1)
    r_s2 = smooth_ramp(strain - b2)
    r_s3 = smooth_ramp(strain - b3)
    
    # Temperature ramp for non-linear temperature dependence
    r_t1 = smooth_ramp(temp - t1)
    
    # Base linear terms
    pred = a0 + a1 * strain + a2 * temp
    
    # Add piecewise strain contributions
    pred = pred + a3 * r_s1 + a4 * r_s2 + a5 * r_s3
    
    # Add temperature non-linearity
    pred = pred + a2 * r_t1
    
    return pred
# EVOLVE-BLOCK-END
