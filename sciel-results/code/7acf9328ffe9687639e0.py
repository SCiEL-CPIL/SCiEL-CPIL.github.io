"""Cost-aware blended acquisition (EI/cost + bounded UCB) with deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: the evaluator kills after 330 s wall-clock.
# We keep a generous safety margin and a hard cap per acquisition call.
_DEADLINE_START = time.time()
# 330 s total budget; leave ~60 s margin for evaluator overhead / other instances.
_HARD_LIMIT = 330.0 - 60.0  # seconds of our own work we allow to elapse
# Per-call soft cap (the acquisition itself should be milliseconds, but guard anyway).
_CALL_SOFT_LIMIT = 5.0


def _safe_z(mu, sigma, f_best):
    """z = (f_best - mu) / sigma with sigma floored to avoid div-by-zero."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    # Clip to a numerically safe range for cdf/pdf.
    return np.clip(z, -1e4, 1e4)


def acquisition(mu, sigma, f_best, cost):
    """Return 2000 finite acquisition scores (higher = evaluate next).

    Blends cost-normalized Expected Improvement (exploitation) with a bounded
    Upper Confidence Bound term (exploration). Cost normalization keeps the
    behavior aligned with the cost-aware objective; the small UCB term guards
    against premature exploitation of noisy low-mu regions.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Deadline guard: if we've burned through our allowed time, return a
    # safe, finite fallback immediately rather than risk the kill.
    if (time.time() - _DEADLINE_START) > _HARD_LIMIT:
        # Fallback: prefer low mu, low cost, high sigma (exploration) — finite.
        fallback = -mu / (np.maximum(cost, 1e-6)) + 0.1 * sigma
        return np.nan_to_num(fallback, nan=0.0, posinf=0.0, neginf=0.0)

    # Cost normalization: cost is (1 + mean(x in [0,1])) in [1, 2], so >= 1.
    cost_safe = np.maximum(cost, 1e-6)

    # Expected Improvement for minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = _safe_z(mu, sigma, f_best)
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = (f_best - mu) * cdf + sigma * pdf
    # EI can be slightly negative when mu > f_best; clamp to >= 0 for a score.
    ei = np.maximum(ei, 0.0)
    ei_cost = ei / cost_safe

    # Bounded exploration term: sigma / cost_safe, scaled small so it can't
    # dominate the exploitation signal.
    explore = (sigma / cost_safe) * 0.05

    score = ei_cost + explore

    # Guarantee all-finite output.
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    return score
# EVOLVE-BLOCK-END
