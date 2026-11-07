"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration with:
    - Linear and nonlinear stiffness (x, x^3)
    - Linear and nonlinear damping (v, v^3)
    - Time-dependent sinusoidal driving force
    - Cross terms and constant offset
    
    Uses 10 parameters:
    p0: linear stiffness coefficient
    p1: cubic stiffness coefficient  
    p2: linear damping coefficient
    p3: cubic damping coefficient
    p4: driving amplitude
    p5: driving frequency
    p6: driving phase
    p7: cross term x*v coefficient
    p8: x*t coupling coefficient
    p9: constant offset
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Nonlinear stiffness terms
    stiffness = p0 * x + p1 * x**3
    
    # Nonlinear damping terms
    damping = p2 * v + p3 * v**3
    
    # Time-dependent driving force (sinusoidal)
    driving = p4 * np.sin(p5 * t + p6)
    
    # Cross coupling terms
    cross = p7 * x * v + p8 * x * t
    
    # Combine all terms
    acceleration = stiffness + damping + driving + cross + p9
    
    return acceleration
# EVOLVE-BLOCK-END
