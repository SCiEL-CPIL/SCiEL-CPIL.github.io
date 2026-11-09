"""
Symbolic regression for stress-strain of an aluminium rod.
Uses a flexible piecewise-linear model with three ReLU hinges (knots)
and temperature-dependent slopes and knot positions for BFGS fitting.
"""
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Model structure:
    - Smooth piecewise-linear approximation using ReLU hinge functions
    - Three knots: yield point, initial hardening transition, secondary transition
    - Temperature-dependent knot positions and slopes
    - Quadratic temperature correction for residual curvature
    
    Parameters:
    p0: intercept (base stress offset)
    p1: elastic slope base (Young's modulus at ref temp)
    p2: elastic slope temperature coefficient
    p3: first knot (yield strain) base
    p4: first knot temperature coefficient
    p5: second knot (hardening transition) base
    p6: second knot temperature coefficient
    p7: slope increment after first knot (initial hardening) base
    p8: slope increment after second knot base
    p9: slope increment after third knot (secondary hardening)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Center temperature for better conditioning
    tc = temp - 0.5
    
    # Temperature-dependent elastic slope
    E = p1 + p2 * tc
    
    # Temperature-dependent knot positions
    k1 = p3 + p4 * tc       # yield strain
    k2 = p5 + p6 * tc       # hardening transition strain
    k3 = 0.5                # fixed third knot (mid-range for secondary hardening)
    
    # ReLU hinge functions
    h1 = np.maximum(strain - k1, 0.0)
    h2 = np.maximum(strain - k2, 0.0)
    h3 = np.maximum(strain - k3, 0.0)
    
    # Temperature-dependent slope increments
    S1 = p7 + 0.5 * p2 * tc   # initial hardening slope (coupled to E temp coeff)
    S2 = p8 + 0.3 * p2 * tc   # secondary hardening slope
    S3 = p9                   # tertiary hardening slope (constant)
    
    # Construct stress: linear + sum of hinge contributions
    stress = p0 + E * strain + S1 * h1 + S2 * h2 + S3 * h3
    
    # Small quadratic temperature correction for residual curvature
    stress = stress + 0.1 * p2 * tc * tc * strain
    
    return stress
