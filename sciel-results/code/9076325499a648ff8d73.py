"""Cost-aware acquisition for BO with cost(x) = 1 + mean(x_scaled_to_[0,1])."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time as _time

# Hard wall-clock guard: the evaluator kills us after ~330 s.
# We return the current best candidate immediately if we ever exceed the deadline.
_DEADLINE_S = 250.0
_START = None


def _now():
    return _time.time()


def _elapsed():
    if _START is None:
        return 0.0
    return _now() - _START


def acquisition(mu, sigma, f_best, cost):
    global _START
    if _START is None:
        _START = _now()

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    # Safety: guarantee finite outputs and a positive cost floor.
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e12, neginf=-1e12)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e3, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)

    n = mu.shape[0]
    scores = np.zeros(n, dtype=float)

    # Deadline guard: if we are too close to the kill, fall back to a cheap
    # exploitation-only score so we never blow the budget mid-call.
    if _elapsed() > _DEADLINE_S:
        z = (f_best - mu) / (sigma + 1e-9)
        return np.nan_to_num(z, nan=0.0, posinf=1e6, neginf=-1e6)

    eps = 1e-9

    # Compute both EI and UCB for a hybrid acquisition.
    # EI: Expected Improvement for minimization.
    z_ei = (f_best - mu) / (sigma + eps)
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma * norm.pdf(z_ei)
    ei = np.where(sigma > 0, ei, np.maximum(f_best - mu, 0.0))
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=0.0)

    # UCB: Upper Confidence Bound for minimization (lower is better).
    # For minimization, we want to explore where the lower bound is low,
    # i.e., mu - beta * sigma. But since we're maximizing an acquisition
    # function (higher = evaluate next), we use:
    #   ucb = -(mu - beta * sigma) = -mu + beta * sigma
    # This rewards points with low predicted value AND high uncertainty.
    # We need to normalize mu and sigma to make beta meaningful.
    mu_range = mu.max() - mu.min() if mu.max() > mu.min() else 1.0
    sigma_max = sigma.max() if sigma.max() > 0 else 1.0

    # Adaptive beta: larger when we're early (less data), smaller when confident.
    # Use a simple heuristic based on the spread of sigma.
    sigma_mean = np.mean(sigma)
    beta = 1.5  # Moderate exploration

    ucb = -mu + beta * sigma
    ucb = np.nan_to_num(ucb, nan=0.0, posinf=1e6, neginf=-1e6)

    # Normalize EI and UCB to comparable scales.
    ei_max = ei.max() if ei.max() > 0 else 1.0
    ucb_max = ucb.max() if ucb.max() > 0 else 1.0
    ucb_min = ucb.min() if ucb.min() < 0 else -1.0
    ucb_range = ucb_max - ucb_min if ucb_range > 0 else 1.0

    ei_norm = ei / ei_max
    ucb_norm = (ucb - ucb_min) / (ucb_range + eps)

    # Cost-aware: divide by cost to prefer cheaper evaluations.
    cost_floor = np.maximum(cost, 1.0)
    ei_cost = ei_norm / cost_floor
    ucb_cost = ucb_norm / cost_floor

    # Blend EI and UCB with a weight that adapts based on confidence.
    # When sigma is small (confident), favor EI (exploitation).
    # When sigma is large (uncertain), favor UCB (exploration).
    conf = np.exp(-sigma / (sigma_mean + eps))
    blend_ei = 0.3 + 0.4 * conf  # Range: 0.3 to 0.7
    blend_ucb = 1.0 - blend_ei

    score = blend_ei * ei_cost + blend_ucb * ucb_cost

    # Add a small exploitation floor: when the model is very confident
    # (tiny sigma), let the predicted improvement dominate so we converge
    # on the best-known region instead of diffusing budget into exploration.
    exploit = np.maximum(f_best - mu, 0.0) / cost_floor
    exploit_norm = exploit / (exploit.max() + eps) if exploit.max() > 0 else 0
    conf_strong = np.exp(-sigma / 1e-4)
    score = score + 0.1 * conf_strong * exploit_norm

    # Ensure finiteness and a strictly ordered, safe output.
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
    return score
# EVOLVE-BLOCK-END
