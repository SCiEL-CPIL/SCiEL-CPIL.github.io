"""Cost-aware acquisition for Bayesian optimization (styblinski tang 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    # Guard against non-finite / degenerate inputs.
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-3)

    f_best = float(np.asarray(f_best, dtype=float))

    # Expected improvement (higher is better) under a Gaussian posterior.
    z = (f_best - mu) / sigma
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = (f_best - mu) * cdf + sigma * pdf
    ei = np.where(np.isfinite(ei), ei, 0.0)
    ei = np.maximum(ei, 0.0)

    # Cost-aware: reward improvement per unit of true evaluation cost.
    # The known cost structure (cost = 1 + mean(scaled x)) means cheap
    # candidates (lower coordinates) are favored, which is exactly what
    # dividing by the candidate's own cost achieves.
    score = ei / cost

    # Small exploration bonus in low-uncertainty-improvement regions so the
    # search does not collapse entirely onto the single best-improvement point.
    score = score + 1e-3 * sigma

    # Ensure a finite, well-ordered output.
    score = np.where(np.isfinite(score), score, 0.0)
    return score
# EVOLVE-BLOCK-END
