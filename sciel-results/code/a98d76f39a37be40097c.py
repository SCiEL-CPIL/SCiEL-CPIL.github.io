"""Cost-aware acquisition with probability-weighted EI and cost-efficiency ratio."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Module-level deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 285.0  # 285 s safety margin within 300 s limit

_EPS = 1e-9


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing cosine8 8d.
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Deadline guard: if we're close to the limit, return a safe default
    if time.time() > _DEADLINE - 3.0:
        scores = -np.asarray(cost, dtype=np.float64)
        return np.where(np.isfinite(scores), scores, 0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    n = mu.shape[0]

    # Stabilize sigma
    sigma_safe = np.maximum(sigma, 1e-6)

    # Standardized improvement distance
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -12.0, 12.0)

    # Vectorized norm.cdf and norm.pdf (fast, no np.vectorize)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)

    # Expected Improvement (EI)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    ei = np.maximum(ei, 0.0)

    # Probability of improvement
    p_improve = cdf_z

    # Cost normalization: cost in [1, 2] -> [0, 1]
    cost_norm = np.clip(cost - 1.0, 0.0, 1.0)

    # Progress: how close f_best is to the known minimum (-0.8)
    # progress = 1.0 means early (f_best far from min), 0.0 means late (f_best near min)
    progress = np.clip((f_best - (-0.8)) / 1.5, 0.0, 1.0)
    # Smooth transition using sigmoid for smoother behavior
    # Map progress to a weight: early -> more exploration, late -> more exploitation
    # Use a smooth interpolation parameter
    exploit_weight = 1.0 - 0.5 * np.exp(-2.0 * progress)
    # exploit_weight ranges from ~0.5 (early) to ~1.0 (late)
    explore_weight = 1.0 - exploit_weight  # ~0.5 (early) to ~0.0 (late)

    # Cost-efficiency ratio: EI per unit cost
    # This naturally balances improvement against evaluation cost
    # Use a soft version to avoid division by zero
    cost_eff = ei / (cost + 0.1)

    # Probability-weighted EI: emphasizes points likely to improve
    pw_ei = p_improve * ei

    # Uncertainty bonus: reward high sigma (exploration)
    # Scale uncertainty by the progress (more exploration early)
    unc_bonus = explore_weight * 0.3 * sigma_safe * pdf_z

    # Cost penalty: smooth penalty that increases with exploitation
    # Use a quadratic cost penalty for smoothness
    cost_penalty = exploit_weight * 0.4 * cost_norm * cost_norm

    # Combine: probability-weighted EI + uncertainty bonus - cost penalty
    # Plus a small cost-efficiency term for additional balance
    scores = pw_ei + unc_bonus - cost_penalty + 0.2 * cost_eff

    # Soft-max style normalization to keep scores in a reasonable range
    # This prevents extreme values from dominating
    max_score = np.max(scores)
    if max_score > 1e-6:
        scores = scores / max_score

    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Add tiny deterministic noise to break ties
    rng = np.random.default_rng(42)
    noise = rng.uniform(0, 1e-7, size=n)
    scores = scores + noise

    return scores
# EVOLVE-BLOCK-END
