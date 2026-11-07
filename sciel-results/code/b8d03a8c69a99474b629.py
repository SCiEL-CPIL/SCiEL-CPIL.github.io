"""Symbolic regression: nonlinear oscillator (2) - closed-form equation."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator with time-dependent driving force.
    
    Model: a = p0*x + p1*v + p2*x^3 + p3*v^3 + p4*sin(p5*t + p6) + p7*t + p8*x*v + p9*t*x
    
    Physics-motivated terms:
    - p0*x: linear restoring force
    - p1*v: linear damping
    - p2*x^3: nonlinear (Duffing) stiffness
    - p3*v^3: nonlinear damping
    - p4*sin(p5*t + p6): periodic driving force
    - p7*t: linear time drift
    - p8*x*v: state cross-term
    - p9*t*x: time-state interaction
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Linear dynamics
    linear = p0 * x + p1 * v
    
    # Nonlinear stiffness and damping
    nonlinear_state = p2 * x**3 + p3 * v**3
    
    # Periodic driving force
    driving = p4 * np.sin(p5 * t + p6)
    
    # Time-dependent terms
    time_terms = p7 * t + p9 * t * x
    
    # Cross term
    cross = p8 * x * v
    
    return linear + nonlinear_state + driving + time_terms + cross
# EVOLVE-BLOCK-END
