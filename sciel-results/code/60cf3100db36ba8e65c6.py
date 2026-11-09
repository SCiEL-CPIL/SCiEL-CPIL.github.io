"""Rewritten construct() for the Laguerre double-roots uncertainty task.

Strategy: place the 12 double roots at the positive zeros of the Bessel
function J_{1/2}, i.e. x_i = (pi*i)**2 for i = 1..12. These are the exact
roots of sin(sqrt(x))/sqrt(x), which matches the asymptotic/orthogonal
structure of the generalized Laguerre polynomials L_n^{(-1/2)}(x).
"""
import math
import time
import random

_DEADLINE_SECONDS = 590.0  # safety margin before the 630s kill
_START_TIME = time.time()


def _remaining():
    return _DEADLINE_SECONDS - (time.time() - _START_TIME)


def _bessel_half_zeros(count):
    """Positive zeros of J_{1/2}(sqrt(x)): x_i = (pi*i)^2, i=1..count."""
    return [(math.pi * i) ** 2 for i in range(1, count + 1)]


def construct():
    # Primary candidate: exact Bessel J_{1/2} zeros (sin(sqrt(x))/sqrt(x) roots).
    primary = _bessel_half_zeros(12)

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
        # Fallback to the seed spacing (still a valid finite, positive, distinct set).
        return [4.0 * (i + 1) for i in range(12)]

    # Light local refinement: if we have budget, try a small number of
    # geometrically-scaled perturbations of the Bessel zeros. This is a cheap
    # heuristic; the exact Bessel zeros are the main bet.
    best = primary
    best_key = None

    def _score_key(zs):
        # Heuristic surrogate: prefer configurations whose spacing grows
        # smoothly (Bessel zeros grow ~ i^2). Penalize irregular ratios.
        s = sorted(zs)
        ratios = [s[i + 1] / s[i] for i in range(len(s) - 1)]
        # Ideal ratio for x_i ~ i^2 is (i+1)^2/i^2 ~ 1 + 2/i.
        ideal = [((i + 2) ** 2) / ((i + 1) ** 2) for i in range(len(ratios))]
        return sum((r - ir) ** 2 for r, ir in zip(ratios, ideal))

    best_key = _score_key(best)

    # A few cheap perturbations if time allows.
    if _remaining() > 30.0:
        rng = random.Random(12345)
        for _ in range(40):
            if _remaining() < 20.0:
                break
            cand = []
            for i in range(1, 13):
                base = (math.pi * i) ** 2
                # small multiplicative jitter
                jitter = 1.0 + rng.uniform(-0.02, 0.02)
                cand.append(base * jitter)
            if not _valid(cand):
                continue
            k = _score_key(cand)
            if k < best_key:
                best_key = k
                best = cand

    return best
