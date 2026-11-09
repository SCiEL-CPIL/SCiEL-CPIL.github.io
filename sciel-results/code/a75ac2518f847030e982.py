"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Acceleration of a damped nonlinear oscillator with driving force, from position and velocity."""
    # params: p0..p9 (length 10)
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Stiffness term: Linear + Nonlinear saturation
    # Uses tanh to bound the nonlinearity, improving conditioning
    # Structure: k_eff = p1 + p2 * tanh(p3 * x^2 + p4 * v)
    stiffness = p1 + p2 * np.tanh(p3 * x**2 + p4 * v)

    # Damping term: Linear + Nonlinear saturation
    # Structure: c_eff = p5 + p6 * tanh(p7 * x + p8 * v)
    damping = p5 + p6 * np.tanh(p7 * x + p8 * v)

    # Constant offset / driving bias
    offset = p9

    # Acceleration = -stiffness * x - damping * v + offset
    acc = -stiffness * x - damping * v + offset

    return acc
# EVOLVE-BLOCK-END
