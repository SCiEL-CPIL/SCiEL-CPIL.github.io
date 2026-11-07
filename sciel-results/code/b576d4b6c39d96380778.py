"""Cost-aware acquisition function for Bayesian optimization (4 test functions)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function for cost-aware Bayesian optimization.

    Parameters:
    - mu: array of shape (2000,) - GP mean predictions
    - sigma: array of shape (2000,) - GP standard deviations
    - f_best: float - best observed value so far
    - cost: array of shape (2000,) - evaluation costs for each candidate

    Returns:
    - array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    t_start = time.time()

    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    n = len(mu)
    if n == 0:
        return np.array([])

    # Safety: clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)

    # Compute z-scores for Expected Improvement
    z = (f_best - mu) / sigma_safe

    # Expected Improvement (EI) for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Probability of Improvement (PI)
    pi = norm.cdf(z)

    # Upper Confidence Bound (UCB) - exploration
    # Adaptive kappa: start high for exploration, decay toward exploitation
    # Budget remaining estimate: total budget is 30, we don't know current total cost
    # but we can use the mean cost as a proxy for how "expensive" candidates are
    mean_cost = np.mean(cost_safe)
    # Estimate remaining budget fraction: assume we've used some fraction
    # Use a heuristic: if mean cost is low, we're in early stages (explore more)
    # If mean cost is high, we're in later stages (exploit more)
    # Scale: cost ranges from 1 to 2 (since 1 + mean(x_scaled) where x_scaled in [0,1])
    # So cost in [1, 2]. Normalize: (cost - 1) / 1 = cost - 1 in [0, 1]
    cost_normalized = np.clip((cost_safe - 1.0) / 1.0, 0.0, 1.0)
    mean_cost_norm = np.mean(cost_normalized)

    # Adaptive kappa: high early (mean_cost_norm low), low late (mean_cost_norm high)
    # kappa ranges from 3.0 (exploration) to 0.5 (exploitation)
    kappa = 3.0 - 2.5 * mean_cost_norm

    # UCB for minimization: lower is better, so we want high mu - kappa*sigma
    ucb = mu - kappa * sigma_safe
    # Convert to improvement-like score: higher is better
    ucb_score = -ucb

    # Normalize each component to [0, 1] for blending
    def normalize(a):
        mn = np.min(a)
        mx = np.max(a)
        rng = mx - mn
        if rng < 1e-12:
            return np.zeros_like(a)
        return (a - mn) / rng

    ei_norm = normalize(ei)
    pi_norm = normalize(pi)
    ucb_norm = normalize(ucb_score)

    # Structural change: use a cost-aware blend where the weights adapt
    # to the relative cost of each candidate. Cheaper candidates get a
    # slightly higher weight on exploitation (EI), expensive ones get
    # more weight on exploration (UCB) to justify the cost.
    # cost_weight: in [0, 1], higher means more expensive
    cw = cost_normalized

    # Weight for EI: higher for cheap candidates
    w_ei = 0.5 + 0.3 * (1.0 - cw)
    # Weight for PI: moderate
    w_pi = 0.15
    # Weight for UCB: higher for expensive candidates
    w_ucb = 0.35 * cw + 0.15

    # Normalize weights to sum to 1
    w_sum = w_ei + w_pi + w_ucb
    w_ei /= w_sum
    w_pi /= w_sum
    w_ucb /= w_sum

    # Combined score before cost penalty
    combined = w_ei * ei_norm + w_pi * pi_norm + w_ucb * ucb_norm

    # Cost-aware penalty: divide by cost^0.5 (geometric mean penalty)
    # This is a structural choice: penalize expensive evaluations
    # but not too harshly (exponent 0.5 vs 1.0)
    cost_penalty = np.sqrt(cost_safe)
    scores = combined / cost_penalty

    # Add a small uncertainty bonus to encourage exploration in high-sigma regions
    # This is scaled by 1/cost to avoid rewarding expensive uncertain points
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-12:
        uncertainty_bonus = 0.05 * (sigma_safe / sigma_max) / cost_safe
    else:
        uncertainty_bonus = np.zeros_like(scores)
    scores += uncertainty_bonus

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Deadline guard: return immediately if we're close to the limit
    elapsed = time.time() - t_start
    if elapsed > 320.0:
        return scores

    return scores
# EVOLVE-BLOCK-END
