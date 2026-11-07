"""Stress-strain model for aluminium rod using smooth hinge yield transition."""
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Model: stress = E(T) * strain + H(T) * h(strain - eps_y(T))
    where h(x) = x * (1 - exp(-k * x)) for x > 0 and 0 otherwise.
    This creates a smooth transition from elastic slope E to plastic slope E + H.
    
    Uses 10 parameters:
    p0: base elastic modulus E0
    p1: temperature coefficient for E
    p2: base hardening modulus H0
    p3: temperature coefficient for H
    p4: base yield strain eps_y0
    p5: temperature coefficient for eps_y
    p6: base sharpness k0
    p7: temperature coefficient for k
    p8: temperature offset
    p9: constant offset
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Temperature-dependent elastic modulus
    E = p0 + p1 * temp
    
    # Temperature-dependent hardening modulus
    H = p2 + p3 * temp
    
    # Temperature-dependent yield strain
    eps_y = p4 + p5 * temp
    
    # Temperature-dependent sharpness
    # Ensure k is positive to avoid overflow/instability in exp
    k = np.abs(p6 + p7 * temp) + 1e-6
    
    # Plastic strain part: only active when strain > eps_y
    x = strain - eps_y
    x_pos = np.maximum(x, 0.0)
    
    # Smooth hinge function: x * (1 - exp(-k*x))
    # For small x, approx k*x^2. For large x, approx x.
    # This is stable for x_pos >= 0 and k > 0.
    hinge_val = x_pos * (1.0 - np.exp(-k * x_pos))
    
    # Combine terms
    stress = E * strain + H * hinge_val + p8 * temp + p9
    
    return stress
