"""Symbolic regression: nonlinear oscillator acceleration."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form model for the acceleration of a damped nonlinear oscillator.
    
    Basis functions:
    1. x (linear restoring)
    2. x^3 (cubic restoring, Duffing)
    3. v (linear damping)
    4. v^3 (cubic damping)
    5. x^2 * v (Van der Pol damping)
    6. sin(t) (sinusoidal drive)
    7. cos(t) (cosinusoidal drive / phase shift)
    8. x * sin(t) (parametric excitation)
    9. v * sin(t) (velocity-dependent drive modulation)
    10. 1 (offset)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Position terms
    x_linear = p0 * x
    x_cubic = p1 * x**3

    # Velocity terms
    v_linear = p2 * v
    v_cubic = p3 * v**3

    # Interaction terms
    vd_pol = p4 * x**2 * v

    # Time-dependent drive terms
    time_sin = p5 * np.sin(t)
    time_cos = p6 * np.cos(t)
    param_exc = p7 * x * np.sin(t)
    vel_mod = p8 * v * np.sin(t)

    # Offset
    offset = p9

    return x_linear + x_cubic + v_linear + v_cubic + vd_pol + time_sin + time_cos + param_exc + vel_mod + offset
# EVOLVE-BLOCK-END
