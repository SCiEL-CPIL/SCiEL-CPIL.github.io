"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature."""
    # params: length 10
    # p0: elastic modulus base
    # p1: plastic hardening coefficient
    # p2: yield stress offset
    # p3: temperature sensitivity
    # p4: nonlinearity exponent
    # p5: thermal expansion coefficient
    # p6: interaction term coefficient
    # p7: saturation limit
    # p8: strain rate-like correction (using strain^3)
    # p9: residual stress

    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Ensure strain is non-negative for physical stress-strain
    # But we can have negative strain (compression)
    # Use a smooth model that handles both tension and compression

    # Elastic region: linear
    # Plastic region: nonlinear hardening
    # Temperature dependence: reduces modulus and yield strength

    # Effective strain (accounting for thermal effects)
    # Thermal strain: alpha * (T - T_ref)
    # We'll model temp as a normalized temperature effect

    # Base elastic response
    elastic = p0 * strain * (1.0 - p3 * temp)

    # Plastic hardening: use a smooth saturation model
    # hardening = p1 * (1 - exp(-p4 * abs(strain))) * sign(strain)
    # But we need to be careful with sign and smoothness
    # Use: p1 * strain * exp(-p4 * abs(strain)) for a decaying hardening
    # Or: p1 * (abs(strain) ** p4) * sign(strain) for power-law hardening

    # Let's use a combination:
    # 1. Linear elastic term with temp correction
    # 2. Quadratic plastic term
    # 3. Saturation term
    # 4. Interaction terms

    # Model:
    # stress = p0 * strain * (1 - p3 * temp)
    #        + p1 * strain^2 * (1 - p3 * temp)
    #        + p2 * (1 - p3 * temp)
    #        + p5 * temp * strain
    #        + p6 * temp
    #        + p7 * strain^3
    #        + p8 * temp^2
    #        + p9

    # But this is just a polynomial. Let's make it more physical.

    # Physical model:
    # 1. Elastic modulus E(T) = p0 * (1 - p3 * temp)
    # 2. Yield stress sigma_y(T) = p2 * (1 - p3 * temp)
    # 3. Hardening: sigma = sigma_y + p1 * (strain - sigma_y/E) ** p4
    #    for strain > sigma_y/E, else sigma = E * strain

    # This is complex. Let's use a smoother approximation.

    # Use a hyperbolic tangent to smoothly transition from elastic to plastic
    # Or use a rational function

    # Let's try:
    # stress = (p0 * strain * (1 - p3 * temp) + p2 * (1 - p3 * temp)) / (1 + p1 * abs(strain) * (1 - p3 * temp))
    # This gives a saturation behavior

    # Or use a simpler but effective form:
    # stress = p0 * strain * (1 - p3 * temp) * (1 / (1 + p1 * abs(strain)))
    #        + p2 * (1 - p3 * temp) * tanh(p4 * strain)
    #        + p5 * temp
    #        + p6 * strain * temp
    #        + p7 * strain^2
    #        + p8 * temp^2
    #        + p9

    # Let's implement this:
    strain_abs = np.abs(strain)
    strain_sq = strain * strain
    temp_sq = temp * temp
    strain_temp = strain * temp

    # Elastic term with temperature correction
    elastic_term = p0 * strain * (1.0 - p3 * temp)

    # Saturation factor
    sat_factor = 1.0 / (1.0 + p1 * strain_abs)

    # Plastic/hardening term using tanh for smoothness
    plastic_term = p2 * (1.0 - p3 * temp) * np.tanh(p4 * strain)

    # Temperature-dependent terms
    temp_linear = p5 * temp
    interaction = p6 * strain_temp
    strain_quad = p7 * strain_sq
    temp_quad = p8 * temp_sq
    residual = p9

    # Combine
    stress = elastic_term * sat_factor + plastic_term + temp_linear + interaction + strain_quad + temp_quad + residual

    return stress
# EVOLVE-BLOCK-END
