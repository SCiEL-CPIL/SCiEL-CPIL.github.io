# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for aluminium rod.
    Uses a multiplicative temperature-modulated elastic-plastic model with
    smooth saturation via tanh to capture hardening.
    """
    eps = strain
    T = temp

    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Temperature-dependent elastic modulus: E(T) = p0 + p1*T + p2*T^2
    E_T = p0 + p1 * T + p2 * T * T

    # Temperature-dependent yield stress: sigma_y(T) = p3 + p4*T
    sigma_y = p3 + p4 * T

    # Strain hardening term: use tanh for smooth saturation
    # alpha controls the rate of hardening
    alpha = p5 + p6 * T

    # Hardening factor: sinh-like but bounded via tanh
    # Use tanh to ensure bounded growth
    hardening = np.tanh(alpha * eps)

    # Elastic contribution
    elastic = E_T * eps

    # Plastic contribution (active after yield, smoothed)
    # Use a smooth step to transition from elastic to plastic
    # plastic_factor = tanh((eps - eps_yield) * beta) where eps_yield is related to sigma_y/E_T
    eps_yield = sigma_y / (E_T + 1e-10)
    beta = p7 + p8 * T
    plastic_factor = np.tanh(beta * (eps - eps_yield))

    # Ensure plastic factor is non-negative
    plastic_factor = np.maximum(plastic_factor, 0.0)

    # Plastic stress contribution
    plastic_stress = sigma_y * plastic_factor * hardening

    # Combine elastic and plastic contributions
    stress = elastic + plastic_stress

    # Add a small offset
    stress = stress + p9

    return stress
# EVOLVE-BLOCK-END
