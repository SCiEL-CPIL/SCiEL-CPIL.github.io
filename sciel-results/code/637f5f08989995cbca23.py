"""Piecewise-linear sawtooth basis for aluminium stress-strain with temperature."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature.

    Uses a piecewise-linear 'sawtooth' / zigzag basis built from abs and min
    to capture discrete yield points and work-hardening segments.
    """
    s = np.asarray(strain, dtype=float)
    t = np.asarray(temp, dtype=float)
    p = np.asarray(params, dtype=float)

    eps = 1e-6
    # Clamp strain to a small positive lower bound so abs-based kinks are
    # well-defined and gradients stay finite near zero.
    s = np.maximum(s, eps)

    # Temperature-dependent yield threshold:
    # yield strain shifts with temperature (higher temp -> lower yield).
    yield_pt = p[0] + p[1] * t

    # Two sawtooth (zigzag) terms: a coarse one for the main yield kink and
    # a finer one for secondary work-hardening segments.
    coarse_period = p[2] + eps
    fine_period = p[3] + eps

    # Sawtooth: distance to nearest multiple of the period, folded around 0.
    # This produces kinks (piecewise-linear) at each yield/work-hardening point.
    def sawtooth(x, period):
        # fold x into [0, period), then distance to nearest endpoint -> zigzag
        phase = np.mod(x, period)
        # distance from the nearest kink (0 or period)
        return np.minimum(phase, period - phase)

    coarse_zig = sawtooth(s - yield_pt, coarse_period)
    fine_zig = sawtooth(s - yield_pt, fine_period)

    # Base linear elastic response plus the kinked hardening contributions.
    base = p[4] * s + p[5] * t + p[6]

    # The sawtooth terms add kinked, piecewise-linear hardening.
    # Keep magnitudes bounded with min to avoid runaway growth.
    hard_coarse = p[7] * np.minimum(coarse_zig, coarse_period / 2.0)
    hard_fine = p[8] * np.minimum(fine_zig, fine_period / 2.0)

    # A smooth temperature-modulated correction so temp can reshape the curve.
    temp_mod = p[9] * t * s

    return base + hard_coarse + hard_fine + temp_mod
# EVOLVE-BLOCK-END
