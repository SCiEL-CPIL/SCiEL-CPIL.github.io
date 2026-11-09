"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Piecewise elastic-plastic model with smooth yield transition and
    power-law hardening, modulated by temperature.
    """
    e = strain
    t = temp

    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # p0: offset
    # p1: elastic modulus (slope in elastic region)
    # p2: yield strain (transition center)
    # p3: yield width (inverse sharpness of transition)
    # p4: hardening modulus (slope just past yield)
    # p5: hardening saturation scale
    # p6: temperature shift of yield strain
    # p7: temperature modulation of elastic modulus
    # p8: temperature modulation of hardening
    # p9: thermal stress offset

    # Temperature-adjusted yield strain
    yield_strain = p2 + p6 * (t - 0.5)

    # Smooth transition: 0 in elastic, 1 in plastic
    # Use a sigmoid centered at yield_strain with width controlled by p3
    transition = 0.5 * (1.0 + np.tanh(p3 * (e - yield_strain)))

    # Elastic branch: linear
    elastic_mod = p1 + p7 * (t - 0.5)
    elastic_stress = elastic_mod * e

    # Plastic branch: yield offset + hardening
    # Hardening: p4 * (e - yield_strain) * (1 - exp(-p5 * (e - yield_strain)))
    # This gives linear hardening that saturates
    plastic_strain = np.maximum(e - yield_strain, 0.0)
    hardening = p4 * plastic_strain * (1.0 - np.exp(-p5 * plastic_strain))
    hardening_mod = 1.0 + p8 * (t - 0.5)
    plastic_stress = elastic_mod * yield_strain + hardening * hardening_mod

    # Blend elastic and plastic
    stress = (1.0 - transition) * elastic_stress + transition * plastic_stress

    # Thermal stress offset
    thermal = p9 * (t - 0.5)

    return p0 + stress + thermal
# EVOLVE-BLOCK-END
