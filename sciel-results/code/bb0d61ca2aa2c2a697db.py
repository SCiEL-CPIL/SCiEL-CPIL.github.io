# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_DEADLINE = time.time() + 285.0  # safety margin before the 300 s limit


def acquisition(mu, sigma, f_best, cost):
    t0 = time.time()
    if t0 > _DEADLINE:
        return np.zeros_like(mu)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Defensive finiteness
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    sigma = np.maximum(sigma, 1e-9)

    # Soft-thresholded improvement: 1 - exp(-lambda * max(0, f_best - mu))
    # This is bounded in [0, 1), smooth, and less sensitive to outlier GP predictions
    # than raw EI or linear PI. The parameter lambda controls the "sharpness" of the
    # improvement signal.
    improvement = np.maximum(0.0, f_best - mu)
    lambda_param = 1.0 / (np.mean(sigma) + 1e-12)  # adapt to scale of uncertainty
    soft_improve = 1.0 - np.exp(-lambda_param * improvement)

    # Uncertainty confidence term: how much we don't know here
    # Scaled so that high-uncertainty regions get a boost
    sigma_max = np.max(sigma)
    if sigma_max < 1e-12:
        sigma_max = 1.0
    uncertainty = sigma / sigma_max  # in [0, 1]

    # Cost normalization: prefer cheap evaluations
    cost_norm = cost / np.max(cost) if np.max(cost) > 1e-12 else cost

    # Combine: soft improvement drives exploitation, uncertainty drives exploration
    # Both divided by cost to penalize expensive regions
    # Use a geometric-style blend: soft_improve * (1 + uncertainty) / cost
    # This keeps scores bounded and outlier-resistant
    score = soft_improve * (1.0 + uncertainty) / cost_norm

    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
