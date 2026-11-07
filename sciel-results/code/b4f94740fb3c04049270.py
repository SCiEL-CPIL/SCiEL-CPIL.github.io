# EVOLVE-BLOCK-START
import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.

    Model: Elastic + Lorentzian-rational hardening with quadratic temperature dependence.
    sigma = E(T)*strain + H(T) * (|strain|^2) / (a^2 + |strain|^2) + C0 + C1*T

    The Lorentzian term smoothly transitions from ~|strain| (elastic-like) at small strain
    to a saturation plateau at large strain, capturing yield and hardening in one smooth function.

    params:
    [0]: E0  - elastic modulus at ref T
    [1]: E1  - linear T coefficient for E
    [2]: E2  - quadratic T coefficient for E
    [3]: H0  - hardening amplitude at ref T
    [4]: H1  - linear T coefficient for H
    [5]: H2  - quadratic T coefficient for H
    [6]: a   - yield strain scale (Lorentzian width)
    [7]: b   - curvature modifier (shapes the transition)
    [8]: C0  - constant offset
    [9]: C1  - linear temperature offset
    """
    E0, E1, E2 = params[0], params[1], params[2]
    H0, H1, H2 = params[3], params[4], params[5]
    a = params[6]
    b = params[7]
    C0 = params[8]
    C1 = params[9]

    # Ensure a is positive and bounded for numerical stability
    a_safe = np.clip(a, 1e-8, 10.0)
    a2 = a_safe * a_safe

    # Temperature-dependent Young's modulus (quadratic in T)
    E = E0 + E1 * temp + E2 * temp * temp

    # Temperature-dependent hardening amplitude (quadratic in T)
    H = H0 + H1 * temp + H2 * temp * temp

    # Absolute strain
    abs_strain = np.abs(strain)
    abs_strain2 = abs_strain * abs_strain

    # Lorentzian-rational hardening:
    # f(x) = x^2 / (a^2 + x^2)  ->  0 at x=0, approaches 1 as x->inf
    # With b as a curvature modifier:
    # g(x) = x^2 / (a^2 + b * x^2)  ->  0 at x=0, approaches 1/b as x->inf
    # This gives a smooth S-shaped transition from elastic to saturated hardening.
    denom = a2 + b * abs_strain2
    # Guard against zero/negative denominator
    denom_safe = np.where(np.abs(denom) > 1e-12, denom, 1e-12)
    hardening_frac = abs_strain2 / denom_safe

    # Hardening term
    hardening = H * hardening_frac

    # Elastic term
    elastic = E * strain

    # Temperature offset
    temp_offset = C1 * temp

    # Combine
    stress = elastic + hardening + C0 + temp_offset

    return stress
# EVOLVE-BLOCK-END
