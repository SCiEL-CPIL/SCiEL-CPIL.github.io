"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a hyperbolic tangent based smooth transition from elastic to plastic regime
    with temperature-dependent modulus and yield stress.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Temperature-dependent elastic modulus
    E = p0 + p1 * temp

    # Temperature-dependent yield stress
    sigma_y = p2 + p3 * temp

    # Temperature-dependent transition width
    width = p4 + p5 * temp

    # Plastic hardening slope
    H = p6 + p7 * temp

    # Reference strain for transition center
    eps_0 = p8

    # Constant offset
    offset = p9

    # Ensure width is positive to avoid numerical issues
    width_safe = np.maximum(np.abs(width), 1e-8)

    # Smooth transition factor: goes from 1 (elastic) to 0 (plastic) as strain increases past eps_0
    # tanh((eps_0 - strain) / width) -> 1 when strain << eps_0, 0 when strain >> eps_0
    transition = 0.5 * (1 + np.tanh((eps_0 - strain) / width_safe))

    # Elastic stress component
    sigma_elastic = E * strain

    # Plastic stress component: yield stress + hardening * (strain - eps_0)
    sigma_plastic = sigma_y + H * np.maximum(strain - eps_0, 0)

    # Blend between elastic and plastic
    stress = transition * sigma_elastic + (1 - transition) * sigma_plastic

    # Add offset
    stress += offset

    return stress
# EVOLVE-BLOCK-END
