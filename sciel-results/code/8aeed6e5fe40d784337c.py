"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Multiplicative temperature-modulated elastic-plastic model:
      - Temperature-dependent elastic modulus E(T) = p0 * (1 + p1*(T - p2))
      - Yield stress sigma_y(T) = p3 * (1 + p4*(T - p2))
      - Hardening via sinh-based saturation: sigma_h = sigma_y * sinh(a*eps)/sinh(a*eps_sat)
      - Smooth elastic-to-plastic transition via tanh
    """
    _t0 = time.perf_counter()
    _DEADLINE = 140.0

    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)

    # Temperature offset
    dT = temp - p2

    # Temperature-dependent elastic modulus
    E_T = p0 * (1.0 + p1 * dT)
    # Ensure E stays positive
    E_T = np.maximum(E_T, 0.01)

    # Elastic stress
    sigma_elastic = E_T * strain

    # Temperature-dependent yield stress
    sigma_y = p3 * (1.0 + p4 * dT)
    # Ensure yield stress stays positive
    sigma_y = np.maximum(sigma_y, 0.01)

    # Plastic strain (positive part)
    eps_pos = np.maximum(strain, 1e-12)

    # Hardening: sinh-based saturation model
    # sigma_plastic = sigma_y * sinh(p5 * eps) / sinh(p5 * p6)
    # p5: hardening rate, p6: saturation strain
    # Clip arguments to avoid overflow in sinh
    arg_num = np.clip(p5 * eps_pos, -50, 50)
    arg_den = np.clip(p5 * p6, -50, 50)
    
    sinh_num = np.sinh(arg_num)
    sinh_den = np.sinh(arg_den)
    # Avoid division by zero
    sinh_den_safe = np.where(np.abs(sinh_den) < 1e-12, 1e-12, sinh_den)
    
    sigma_plastic = sigma_y * sinh_num / sinh_den_safe

    # Smooth transition: elastic below yield, plastic above
    # p7: transition sharpness, p8: yield strain threshold
    transition = 0.5 * (1.0 + np.tanh(p7 * (strain - p8)))

    # Combined stress: blend elastic and plastic
    sigma = (1.0 - transition) * sigma_elastic + transition * sigma_plastic

    # Additional correction: temperature-strain interaction
    # p9: small correction coefficient
    corr = p9 * strain * dT

    prediction = sigma + corr

    if time.perf_counter() - _t0 > _DEADLINE:
        pass

    return prediction
# EVOLVE-BLOCK-END
