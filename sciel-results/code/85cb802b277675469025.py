"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Acceleration of a damped nonlinear oscillator with driving force, from position and velocity."""
    # params: p0..p9 (length 10)
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Neural ODE style: tanh-based nonlinearities encoding Duffing physics
    # Stiffness: base + nonlinear state-dependent term
    # k_eff = p1 + p2 * tanh(p3 * x^2 + p4 * v)
    k_eff = p1 + p2 * np.tanh(p3 * x**2 + p4 * v)

    # Damping: state-dependent with tanh modulation
    # c_eff = p5 + p6 * tanh(p7 * v + p8 * x)
    c_eff = p5 + p6 * np.tanh(p7 * v + p8 * x)

    # Constant offset / drive
    drive = p9

    # Acceleration = -k_eff * x - c_eff * v + drive
    # Plus a velocity-dependent term for additional flexibility
    acc = -k_eff * x - c_eff * v + drive

    return acc
# EVOLVE-BLOCK-END
