"""Symbolic regression: nonlinear oscillator (2) - closed-form equation."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator with time-dependent driving force.
    
    Uses a rich nonlinear basis with 10 parameters to capture:
    - Linear dynamics (restoring force, damping)
    - Nonlinear stiffness (cubic term)
    - Velocity-dependent nonlinearity
    - Time-dependent driving (sinusoidal and polynomial)
    - Cross terms between position, velocity, and time
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Core linear terms: restoring force and damping
    # a = -k*x - c*v + forcing
    # We use params to allow flexible combinations
    
    # Nonlinear restoring force: x + x^3 (Duffing-like)
    nonlinear_stiffness = x + p0 * x**3
    
    # Linear damping
    linear_damping = p1 * v
    
    # Velocity-dependent nonlinearity (quadratic drag)
    quadratic_drag = p2 * v * np.abs(v)
    
    # Time-dependent driving force: combination of sinusoidal and polynomial
    # sin(omega*t) type forcing
    driving_sin = p3 * np.sin(p4 * t)
    
    # Polynomial time dependence
    driving_poly = p5 * t + p6 * t**2
    
    # Cross terms: position-velocity coupling
    cross_xv = p7 * x * v
    
    # Position-time coupling
    cross_xt = p8 * x * t
    
    # Velocity-time coupling
    cross_vt = p9 * v * t
    
    # Combine all terms
    result = (nonlinear_stiffness + linear_damping + quadratic_drag +
              driving_sin + driving_poly + cross_xv + cross_xt + cross_vt)
    
    return result
# EVOLVE-BLOCK-END
