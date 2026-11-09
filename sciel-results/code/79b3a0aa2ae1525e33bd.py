"""Bootstrap portfolio: model-and-optimize for stress-strain of aluminium rod."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIMEOUT_LIMIT = 140.0

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature.

    Design family: parameterized kink basis (piecewise-linear with smooth kinks)
    + linear elastic term + constant offset.

    The kink locations are parameterized via exponential transform to ensure
    positive values and reasonable starting range for BFGS initialization at 1.
    This allows the model to adapt yield point and hardening onset strains to
    the data, unlike fixed-kink models.

    Model:
      stress = E(T) * strain + sum_i A_i * kink(strain, c_i, w_i) + B(T)

    Parameters:
      p0, p1 : E(T) = p0 + p1*temp (linear in temp)
      p2     : c1 = exp(p2) * 0.01  (first kink location, ~yield strain)
      p3     : c2 = exp(p3) * 0.1   (second kink location, ~hardening onset)
      p4     : c3 = exp(p4) * 0.5   (third kink location, ~saturation)
      p5     : A1 (amplitude of kink 1)
      p6     : A2 (amplitude of kink 2)
      p7     : A3 (amplitude of kink 3)
      p8     : w  (smoothness width for all kinks, scaled)
      p9     : B  (constant offset)

    kink(x, c, w) = 0.5 * ((x - c) + sqrt((x - c)^2 + w^2))
    """
    if time.time() - _START_TIME > _TIMEOUT_LIMIT:
        return np.zeros_like(strain, dtype=np.float64)

    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)

    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Temperature-dependent Elastic Modulus (linear)
    E_T = p0 + p1 * temp

    # Parameterized kink locations (exponential transform ensures positive, reasonable range)
    c1 = np.exp(p2) * 0.01   # ~yield strain
    c2 = np.exp(p3) * 0.1    # ~hardening onset
    c3 = np.exp(p4) * 0.5    # ~saturation region

    # Smoothness width (scaled, keep small for kink-like behavior)
    w = np.exp(p8) * 0.005

    # Kink amplitudes
    A1 = p5
    A2 = p6
    A3 = p7

    # Constant offset
    B = p9

    # --- Term 1: Linear elastic ---
    term_elastic = E_T * strain

    # --- Terms 2-4: Kink basis functions ---
    def kink(x, c, w):
        d = x - c
        return 0.5 * (d + np.sqrt(d**2 + w**2))

    term_kink1 = A1 * kink(strain, c1, w)
    term_kink2 = A2 * kink(strain, c2, w)
    term_kink3 = A3 * kink(strain, c3, w)

    # --- Term 5: Constant offset ---
    term_offset = B

    stress = term_elastic + term_kink1 + term_kink2 + term_kink3 + term_offset

    # Ensure finite output
    stress = np.where(np.isfinite(stress), stress, 0.0)

    return stress
# EVOLVE-BLOCK-END
