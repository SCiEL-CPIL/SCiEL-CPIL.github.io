"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a piecewise-linear 'sawtooth' basis to model yield points and
    work-hardening segments. The basis functions are constructed from
    abs and min operations, creating kinked transitions that BFGS can fit.
    
    params (length 10):
      p0: base linear slope (elastic modulus scaling)
      p1: temperature scaling for elastic part
      p2: offset
      p3: yield strain scale
      p4: yield stress amplitude
      p5: work-hardening slope
      p6: second yield strain scale
      p7: second yield amplitude
      p8: higher-order correction
      p9: temperature interaction with plastic region
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

    # Ensure positive scales for the kink locations to avoid degenerate behavior
    s1 = np.abs(p3) + 1e-6
    s2 = np.abs(p6) + 1e-6

    # Elastic region: linear in strain with temperature modulation
    elastic = p0 * strain + p1 * temp * strain + p2

    # First yield point: sawtooth-like kink at strain = s1
    # min(strain, s1) gives a flat plateau after yield
    # abs(strain - s1) gives a V-shape centered at yield
    yield1_plateau = np.minimum(strain, s1)
    yield1_vshape = np.abs(strain - s1)
    
    # Plastic work-hardening: linear beyond yield
    plastic1 = np.maximum(strain - s1, 0.0)

    # Second yield point at strain = s2
    yield2_plateau = np.minimum(strain, s2)
    yield2_vshape = np.abs(strain - s2)
    plastic2 = np.maximum(strain - s2, 0.0)

    # Combine: elastic + yield effects + work-hardening
    # The sawtooth basis: min gives saturation, abs gives kink, max gives post-yield
    stress = (elastic
              + p4 * (yield1_plateau - s1)          # plateau drop at first yield
              + p5 * plastic1                        # work-hardening after first yield
              + p7 * (yield2_plateau - s2)           # second yield effect
              + p8 * plastic2                        # second work-hardening
              + p9 * temp * plastic1)                # temperature effect in plastic region

    # Subtract the constant offsets from the plateau terms to avoid double-counting
    # Actually, let's restructure more carefully:
    # Base elastic response
    base = p0 * strain + p1 * temp * strain + p2
    
    # First yield: when strain > s1, the response deviates from linear
    # Use a smooth-ish kink: min(strain, s1) saturates, then plastic adds back
    kink1 = p4 * (np.minimum(strain, s1) - strain)  # negative for strain > s1, zero below
    hard1 = p5 * np.maximum(strain - s1, 0.0)       # positive beyond yield
    
    # Second yield
    kink2 = p7 * (np.minimum(strain, s2) - strain)
    hard2 = p8 * np.maximum(strain - s2, 0.0)
    
    # Temperature interaction with plastic deformation
    temp_plastic = p9 * temp * np.maximum(strain - s1, 0.0)

    stress = base + kink1 + hard1 + kink2 + hard2 + temp_plastic

    return stress
# EVOLVE-BLOCK-END
