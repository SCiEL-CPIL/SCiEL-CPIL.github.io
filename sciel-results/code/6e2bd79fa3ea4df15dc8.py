"""E. coli growth rate model: Monod substrate x skewed-Gaussian temp/pH."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model with biologically-motivated multiplicative structure.

    mu = p0 * (s / (p1 + s)) * exp(-0.5 * ((temp - p2) / p3)^2 * asym_temp) * exp(-0.5 * ((pH - p4) / p5)^2 * asym_ph)

    where asym_temp = 1 + p6 * sign(temp - p2) gives a steeper decline above the
    thermal optimum (thermal death sharper than inhibition), and similarly for pH.
    b (population density) enters as a mild linear modulation (p7 * b + p8) to
    capture any density-dependent effect without breaking the multiplicative core.
    p9 is an overall scale offset.

    Parameters (10):
    p0: Maximum growth rate scale
    p1: Monod half-saturation constant (Ks)
    p2: Optimal temperature
    p3: Temperature width (sigma)
    p4: Optimal pH
    p5: pH width (sigma)
    p6: Temperature asymmetry (positive = steeper above optimum)
    p7: b linear coefficient
    p8: b linear offset (should be ~1 at typical b)
    p9: Global additive bias
    """
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]

    # Safe widths to avoid division by zero / overflow
    sig_t = np.maximum(np.abs(p3), 0.5)
    sig_p = np.maximum(np.abs(p5), 0.2)
    ks = np.maximum(np.abs(p1), 0.1)

    # Monod substrate saturation
    monod = s / (ks + s)

    # Skewed Gaussian temperature response
    dt = temp - p2
    asym_t = 1.0 + np.clip(p6, -0.9, 0.9) * np.sign(dt)
    temp_resp = np.exp(-0.5 * (dt / sig_t) ** 2 * asym_t)

    # Skewed Gaussian pH response
    dp = pH - p4
    asym_p = 1.0 + np.clip(p6, -0.9, 0.9) * np.sign(dp)
    ph_resp = np.exp(-0.5 * (dp / sig_p) ** 2 * asym_p)

    # Mild linear modulation by population density b
    b_mod = p7 * b + p8

    pred = p0 * monod * temp_resp * ph_resp * b_mod + p9

    return np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
# EVOLVE-BLOCK-END
