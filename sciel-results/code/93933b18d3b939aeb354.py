"""Rewritten construct() for the Laguerre double-roots uncertainty task.

Strategy: Use the roots of the generalized Laguerre polynomial L_12^{(-1/2)}(x)
as the double-root locations z_i. These are the Gaussian quadrature nodes for
the Laguerre weight, which is a principled choice for finite-dimensional
approximation problems, contrasting with the asymptotic Bessel zeros used
previously.
"""
import math
import time
import random
import numpy as np
from scipy.special import roots_genlaguerre

_DEADLINE_SECONDS = 590.0  # safety margin before the 630s kill
_START_TIME = time.time()


def _remaining():
    return _DEADLINE_SECONDS - (time.time() - _START_TIME)


def construct():
    # Primary candidate: roots of L_12^{(-1/2)}(x)
    # These are the nodes for Gaussian quadrature with weight x^{-1/2} e^{-x}
    try:
        # n=12, alpha=-0.5
        roots, _ = roots_genlaguerre(12, -0.5)
        primary = [float(r) for r in roots]
    except Exception:
        # Fallback to Bessel zeros if scipy fails
        primary = [(math.pi * i) ** 2 for i in range(1, 13)]

    # Validate: finite, > 0, pairwise distinct.
    def _valid(zs):
        if len(zs) != 12:
            return False
        for z in zs:
            if not math.isfinite(z) or z <= 0.0:
                return False
        s = sorted(zs)
        for a, b in zip(s, s[1:]):
            if abs(a - b) < 1e-12:
                return False
        return True

    if not _valid(primary):
        # Fallback to simple spacing
        return [4.0 * (i + 1) for i in range(12)]

    # Return the Laguerre roots directly.
    # Given the high cost of evaluation, we do not attempt local search
    # unless we are confident it will improve. The Laguerre roots are a
    # strong structural candidate.
    return primary
