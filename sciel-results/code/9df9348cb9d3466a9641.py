"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration with rational-basis nonlinearity.
    
    The structure uses rational functions to allow parameters to shape
    state-dependent stiffness and damping, plus cross-coupling terms.
    
    Inputs:
        x: position
        v: velocity
        params: array of 10 fitted constants
    
    Returns:
        Acceleration (one prediction per row)
    """
    p = params
    
    # Precompute powers
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    
    # Rational stiffness: (p0*x + p1*x^3) / (1 + p2*x^2)
    stiff_denom = 1.0 + p[2] * x2
    stiff = (p[0] * x + p[1] * x3) / stiff_denom
    
    # Rational damping: (p3*v + p4*v^3) / (1 + p5*v^2)
    damp_denom = 1.0 + p[5] * v2
    damp = (p[3] * v + p[4] * v3) / damp_denom
    
    # Van der Pol type term: p6 * x^2 * v
    vdp = p[6] * x2 * v
    
    # Cross term: p7 * x * v
    cross = p[7] * x * v
    
    # Velocity-stiffness coupling: p8 * x * v^2
    v_coupling = p[8] * x * v2
    
    # Offset: p9
    offset = p[9]
    
    return stiff + damp + vdp + cross + v_coupling + offset
# EVOLVE-BLOCK-END
