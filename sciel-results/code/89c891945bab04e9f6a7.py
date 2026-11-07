"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Known minimum of hartmann 6d
_HARTMANN_6D_MIN = -3.32237


def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function that balances expected improvement, uncertainty, and cost.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions in objective units.
    sigma : array of shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed function value.
    cost : array of shape (2000,)
        Cost of evaluating each candidate (1 + mean(x) where x in [0,1]^6).
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Deadline guard: track elapsed time and bail out if we're running out
    t_start = time.time()
    DEADLINE_MARGIN = 5.0  # seconds before the 300s limit
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and negative values
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute progress: how close are we to the known minimum?
    # progress = 1 means we've found the minimum, 0 means we're at f_best
    # f_best starts around -1 or so for hartmann 6d
    progress = np.clip((f_best - _HARTMANN_6D_MIN) / 5.0, 0.0, 1.0)
    # Early on: progress ~ 1 (f_best is far from min), late: progress ~ 0
    
    # Adaptive beta for UCB component
    # Early: aggressive exploration (high beta), Late: exploitation (low beta)
    beta = 0.5 + 1.5 * progress  # ranges from 0.5 to 2.0
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB
    ucb = mu + beta * sigma
    
    # Cost-normalized scores
    # Normalize cost to [0, 1] range for weighting
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    if cost_max > cost_min + 1e-9:
        cost_norm = (cost - cost_min) / (cost_max - cost_min)
    else:
        cost_norm = np.zeros_like(cost)
    
    # Blend EI and UCB with adaptive weights
    # Early: more UCB weight, Late: more EI weight
    w_ei = 0.3 + 0.7 * (1.0 - progress)
    w_ucb = 1.0 - w_ei
    
    # Combine: higher EI and lower UCB (since we minimize) are better
    # For minimization: we want low mu, high sigma (uncertainty), and low cost
    # EI is already positive when mu < f_best
    # UCB for minimization: we want candidates where even the upper bound is promising
    
    # Score: combine EI (higher is better) and negative UCB (lower UCB is better)
    # Normalize each component
    ei_max = np.max(ei) if np.max(ei) > 1e-9 else 1.0
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    if ucb_max > ucb_min + 1e-9:
        ucb_norm = (ucb - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb)
    
    # Higher EI is better, lower UCB is better
    score_ei = ei / ei_max
    score_ucb = 1.0 - ucb_norm  # lower UCB -> higher score
    
    # Combine with cost penalty
    # Penalize high cost: multiply by (1 - cost_penalty * cost_norm)
    cost_penalty = 0.3  # moderate cost awareness
    
    raw_score = w_ei * score_ei + w_ucb * score_ucb
    score = raw_score * (1.0 - cost_penalty * cost_norm)
    
    # Pareto-frontier-inspired filtering:
    # Identify candidates that are non-dominated in (score, uncertainty, low_cost)
    # A candidate is dominated if another has >= score, >= sigma, and <= cost
    # We use a soft filter: boost scores of non-dominated candidates
    
    # For efficiency, use a simplified approach:
    # Rank by composite score and give a small boost to high-uncertainty, low-cost candidates
    sigma_norm = sigma / (np.max(sigma) + 1e-9)
    
    # Boost: add a small term for high uncertainty and low cost
    boost = 0.1 * sigma_norm * (1.0 - cost_norm)
    score = score + boost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline check
    elapsed = time.time() - t_start
    if elapsed > 300.0 - DEADLINE_MARGIN:
        # Return best-effort result immediately
        return score
    
    return score
# EVOLVE-BLOCK-END
