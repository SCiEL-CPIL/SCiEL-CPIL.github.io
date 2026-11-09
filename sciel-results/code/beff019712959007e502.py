"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a fixed-basis additive model (linear in params) with:
    - Polynomial strain terms for elastic + strain hardening
    - Tanh terms for yield/saturation behavior
    - Temperature terms for thermal softening
    - Interaction terms for thermo-mechanical coupling
    
    All 10 parameters are fitted by BFGS on a convex-like loss landscape.
    """
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Strain powers (elastic + hardening)
    s1 = strain
    s2 = s1 * s1
    s3 = s2 * s1
    
    # Tanh saturation terms for yield behavior
    # Scale strain to make tanh transition around yield point
    t1 = np.tanh(50.0 * s1)       # sharp transition near yield
    t2 = np.tanh(10.0 * s1)       # broader saturation
    t3 = np.tanh(200.0 * s1)      # very sharp, captures elastic-plastic boundary
    
    # Temperature effects
    # temp is normalized to [0.067, 1.0]
    temp_shifted = temp - 0.5     # center temperature effect
    
    # Build the model: linear combination of basis functions
    # Elastic + hardening (polynomial)
    # + yield saturation (tanh)
    # + temperature effects
    # + interactions
    
    stress = (
        p0 * s1 +                          # linear elastic
        p1 * s2 +                          # quadratic hardening
        p2 * s3 +                          # cubic hardening
        p3 * t1 +                          # primary yield saturation
        p4 * t2 +                          # secondary saturation
        p5 * t3 +                          # sharp elastic-plastic transition
        p6 * temp_shifted +                # linear temperature effect
        p7 * s1 * temp_shifted +           # strain-temperature interaction
        p8 * t1 * temp_shifted +           # yield-temperature interaction
        p9                                  # intercept
    )
    
    return stress
# EVOLVE-BLOCK-END
