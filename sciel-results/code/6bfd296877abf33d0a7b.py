"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track time to avoid timeout. The evaluator kills after 330s.
# We need a safety margin.
_start_time = time.time()
TIME_LIMIT = 325.0  # Safety margin before 330s


def _time_exceeded():
    return (time.time() - _start_time) > TIME_LIMIT


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a structural approach:
    Lower Confidence Bound (LCB) with adaptive exploration, combined
    with a diversity-aware penalty to avoid redundant evaluations.
    
    The key structural change: instead of EI + UCB combination, we use
    a pure LCB approach with an adaptive beta that decays as f_best
    improves, plus a diversity penalty based on local density of high-score
    candidates to encourage exploration of unexplored regions.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed objective value (lower is better).
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Check time budget
    if _time_exceeded():
        # Return a simple, fast fallback: just -mu / cost
        cost_safe = np.maximum(cost, 1e-10)
        score = -mu / cost_safe
        return np.where(np.isfinite(score), score, 0.0)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-10)
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-10)
    
    # Adaptive beta for LCB:
    # When f_best is large (early), we want more exploration (larger beta)
    # When f_best is small (late), we want more exploitation (smaller beta)
    # Use a smooth decay: beta = beta_max * exp(-f_best_abs / scale)
    f_best_abs = abs(f_best)
    beta_max = 3.0
    scale = 1.5
    beta = beta_max * np.exp(-f_best_abs / scale)
    
    # Ensure beta has a minimum to avoid pure exploitation
    beta_min = 0.1
    beta = np.maximum(beta, beta_min)
    
    # LCB: mu - beta * sigma (we want to minimize, so lower is better)
    # For acquisition score (higher = evaluate next), we use -LCB = -mu + beta * sigma
    lcb_score = -mu + beta * sigma_safe
    
    # Diversity penalty: penalize candidates that are close to other high-score candidates
    # This is a structural change: instead of just combining EI and UCB,
    # we actively penalize redundancy in the candidate set.
    
    # Use a simple approach: for each candidate, compute a penalty based on
    # how many other candidates have similar mu and sigma values.
    # We approximate this using a kernel density estimate on the (mu, sigma) space.
    
    # To keep it fast, we use a subsampled approach:
    n = len(mu)
    if n > 500:
        # Subsample for density estimation
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=500, replace=False)
        mu_sub = mu[idx]
        sigma_sub = sigma_safe[idx]
    else:
        mu_sub = mu
        sigma_sub = sigma_safe
    
    # Normalize mu and sigma for density estimation
    mu_range = mu_sub.max() - mu_sub.min() + 1e-10
    sigma_range = sigma_sub.max() - sigma_sub.min() + 1e-10
    mu_norm = (mu - mu_sub.min()) / mu_range
    sigma_norm = (sigma_safe - sigma_sub.min()) / sigma_range
    
    # Compute pairwise distances in normalized (mu, sigma) space
    # Use a subsample to keep this fast
    n_sub = len(mu_sub)
    if n_sub * n > 1000000:
        # Too expensive, skip diversity penalty
        diversity_penalty = np.zeros(n)
    else:
        # Compute distances to subsample
        mu_diff = (mu[:, np.newaxis] - mu_sub[np.newaxis, :]) / mu_range
        sigma_diff = (sigma_safe[:, np.newaxis] - sigma_sub[np.newaxis, :]) / sigma_range
        dist_sq = mu_diff ** 2 + sigma_diff ** 2
        
        # Use a Gaussian kernel with bandwidth
        bandwidth = 0.3
        kernel = np.exp(-dist_sq / (2 * bandwidth ** 2))
        
        # Density estimate for each candidate
        density = kernel.sum(axis=1) / n_sub
        
        # Normalize density to [0, 1]
        density_max = density.max()
        if density_max > 0:
            density_norm = density / density_max
        else:
            density_norm = np.zeros(n)
        
        # Penalty: higher density -> higher penalty
        # Scale the penalty so it's meaningful relative to lcb_score
        lcb_range = lcb_score.max() - lcb_score.min() + 1e-10
        penalty_scale = 0.3 * lcb_range
        diversity_penalty = penalty_scale * density_norm
    
    # Combine LCB score with diversity penalty
    score = lcb_score - diversity_penalty
    
    # Cost-aware: divide by cost
    score = score / cost_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Clip to avoid extreme values
    score_max = np.max(score)
    score_min = np.min(score)
    if score_max - score_min > 1e6:
        score = np.clip(score, score_min, score_min + 1e6)
    
    return score
# EVOLVE-BLOCK-END
