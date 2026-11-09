"""
Symbolic regression for stress-strain of an aluminium rod.
Uses a piecewise-linear approximation with ramp functions for efficient BFGS fitting.
The 10 parameters are fitted by the evaluator using BFGS.
"""
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Model structure:
    - Piecewise-linear approximation using clipped linear terms (ramps)
    - Two kinks (yield and transition) with temperature-dependent locations and slopes
    - Elastic region, initial hardening, and secondary hardening
    
    Parameters:
    p0: intercept
    p1: elastic slope base
    p2: elastic slope temp coefficient
    p3: yield strain base (kink 1)
    p4: yield strain temp coefficient
    p5: initial hardening slope base
    p6: initial hardening slope temp coefficient
    p7: transition strain base (kink 2)
    p8: transition strain temp coefficient
    p9: secondary hardening slope
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Center temperature for better conditioning
    tc = temp - 0.5
    
    # Temperature-dependent elastic slope
    E = p1 + p2 * tc
    
    # Temperature-dependent yield strain (kink 1)
    k1 = p3 + p4 * tc
    
    # Temperature-dependent initial hardening slope
    H1 = p5 + p6 * tc
    
    # Temperature-dependent transition strain (kink 2)
    k2 = p7 + p8 * tc
    
    # Secondary hardening slope (constant)
    H2 = p9
    
    # Ramp functions
    ramp1 = np.maximum(strain - k1, 0.0)
    ramp2 = np.maximum(strain - k2, 0.0)
    
    # Construct stress
    stress = p0 + E * strain + H1 * ramp1 + H2 * ramp2
    
    return stress
