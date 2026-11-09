"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.

    Uses a piecewise smooth basis with two fitted kink locations to
    capture elastic-linear, yield, and work-hardening regimes.

    Model:
      sigma = a0 + a1*eps + a2*soft_kink(eps - k1) + a3*soft_kink(eps - k2)
              + a4*soft_kink((eps - k1)^2) + a5*soft_kink((eps - k2)^2)
              + a6*T + a7*eps*T + a8*soft_kink(eps - k1)*T + a9*soft_kink(eps - k2)*T

    where soft_kink(x) = x * sigmoid(x / s) with a fixed smoothness s.

    Parameters (10 constants):
      p0, p1: kink locations (yield strain, hardening onset)
      p2: smoothness scale for kinks
      p3, p4: linear elastic slope and intercept
      p5, p6: first kink amplitudes (linear + quadratic)
      p7, p8: second kink amplitudes (linear + quadratic)
      p9: temperature offset
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    eps = strain
    T = temp

    # Ensure smoothness scale is positive
    s = np.abs(p2) + 1e-6

    def soft_kink(x):
        """Differentiable kink: x * sigmoid(x/s)"""
        return x * (1.0 / (1.0 + np.exp(-x / s)))

    # Kink basis functions
    x1 = soft_kink(eps - p0)
    x2 = soft_kink(eps - p1)
    x1_sq = soft_kink((eps - p0) ** 2)
    x2_sq = soft_kink((eps - p1) ** 2)

    # Base linear elastic + intercept
    base = p3 * eps + p4

    # Kink contributions (linear and quadratic in offset)
    kink1 = p5 * x1 + p6 * x1_sq
    kink2 = p7 * x2 + p8 * x2_sq

    # Temperature effects
    temp_base = p9 * T
    temp_eps = 0.5 * p9 * eps * T  # small interaction, scaled to avoid dominance
    temp_kink1 = 0.1 * p5 * x1 * T  # temperature-modulated kink response
    temp_kink2 = 0.1 * p7 * x2 * T

    sigma = base + kink1 + kink2 + temp_base + temp_eps + temp_kink1 + temp_kink2

    return sigma
# EVOLVE-BLOCK-END
