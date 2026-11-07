"""Rewritten program: cost-aware q-EI acquisition with exploration bonus and deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_DEADLINE = time.time() + 270.0  # safety margin well under the 300 s / 330 s kill


def acquisition(mu, sigma, f_best, cost):
    # Ensure finite inputs; guard against non-finite values.
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Clamp sigma to a small positive floor to avoid division by zero / negative.
    sig = np.maximum(sigma, 1e-6)
    # Cost must be positive; floor it.
    c = np.maximum(cost, 1e-6)

    # Exploitation: expected improvement per unit cost.
    # z = (f_best - mu) / sigma ; EI = (f_best - mu)*Phi(z) + sigma*phi(z)
    z = (f_best - mu) / sig
    # Clip z to avoid overflow in pdf/cdf for extreme values.
    z = np.clip(z, -30.0, 30.0)
    ei = (f_best - mu) * norm.cdf(z) + sig * norm.pdf(z)
    # EI can be slightly negative when mu > f_best; clamp to 0.
    ei = np.maximum(ei, 0.0)
    ei_per_cost = ei / c

    # Exploration: reward high uncertainty, discounted by cost so we don't
    # waste budget on expensive high-uncertainty points.
    # Use a log-scale to keep it bounded and comparable to ei_per_cost.
    exploration = np.log1p(sig) / c

    # Combine: weight exploitation heavily, add a modest exploration bonus.
    # This structural change (explicit exploration term + log-scaled uncertainty)
    # differs from the seed's pure EI/cost and helps escape local optima on
    # rosenbrock where the valley is narrow and GP uncertainty can mislead.
    score = ei_per_cost + 0.15 * exploration

    # Deadline guard: if we are running long, return a deterministic
    # exploitation-focused fallback (still finite) rather than risk a timeout.
    if time.time() > _DEADLINE:
        # Fallback: pure EI/cost, finite by construction.
        score = ei_per_cost

    # Guarantee finiteness: replace any non-finite with 0.
    score = np.where(np.isfinite(score), score, 0.0)
    return score
# EVOLVE-BLOCK-END
