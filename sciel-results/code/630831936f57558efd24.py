# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature using Voce-type hardening."""
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Temperature-dependent yield stress
    sigma_y = p0 + p1 * temp

    # Temperature-dependent saturation stress
    Q = p2 + p3 * temp

    # Temperature-dependent hardening modulus (ensure positive for stability)
    H = p4 + p5 * temp

    # Avoid division by zero or negative H
    H_safe = np.where(np.abs(H) < 1e-6, 1e-6, H)

    # Effective strain (allow small negative)
    eps = strain

    # Voce exponential hardening
    # For negative H, exp(-eps/H) could be problematic; clip for safety
    exponent = -eps / H_safe
    # Clip exponent to avoid overflow
    exponent = np.clip(exponent, -50, 50)
    exp_term = np.exp(exponent)

    # Main Voce term
    voce = Q * (1.0 - exp_term)

    # Additional polynomial corrections for better fit
    linear = p6 * strain
    quad = p7 * strain * strain
    interaction = p8 * temp * strain
    offset = p9

    stress = sigma_y + voce + linear + quad + interaction + offset

    return stress
# EVOLVE-BLOCK-END
