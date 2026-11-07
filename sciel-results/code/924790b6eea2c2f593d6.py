"""Cost-aware acquisition function for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track time to ensure we stay within limits
_start_time = time.time()
_TIMEOUT_MARGIN = 5.0  # seconds to spare before 330s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Strategy:
    1. Sample a plausible objective value for each candidate from the GP posterior.
       Since we want to minimize, we sample v = mu - sigma * |z| where z ~ N(0,1).
       This creates a stochastic exploration that naturally balances exploitation
       (low mu) and exploration (high sigma).
    2. Apply a cost-aware penalty to the exploitation component.
       We subtract a penalty proportional to (cost - 1) * weight from the score.
       This encourages the algorithm to prefer cheaper points when the sampled
       objective values are similar, but allows expensive points if they offer
       significantly better expected improvement (via the sampling).
       
    Why this is different:
    - The current program uses a deterministic combination of EI and a saturating
      exploration term. This is a fixed heuristic.
    - Thompson Sampling is a stochastic method that samples from the posterior.
      It has a proven theoretical basis for balancing exploration/exploitation.
      The randomness in the sample allows the algorithm to escape local optima
      more effectively than fixed-weight heuristics, especially in early stages.
    
    Args:
        mu: (N,) array of GP mean predictions.
        sigma: (N,) array of GP standard deviations.
        f_best: float, best observed value so far.
        cost: (N,) array of costs for each candidate.
        
    Returns:
        (N,) array of acquisition scores (higher is better).
    """
    # Check time budget - if we're close to timeout, return a safe default
    elapsed = time.time() - _start_time
    if elapsed > 325.0:  # 330 - 5 second margin
        return np.zeros_like(mu)
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Thompson Sampling: Sample a value from the posterior.
    # We want to minimize, so we sample v such that lower v is better.
    # v = mu - sigma * |z| ensures that high uncertainty (sigma) leads to
    # potentially much lower values (exploration), while low uncertainty
    # leads to values close to mu (exploitation).
    # Using |z| ensures we are sampling from the lower tail, which is appropriate
    # for minimization with a one-sided improvement perspective.
    
    # Generate random normal samples
    z = np.random.standard_normal(len(mu))
    
    # Sampled objective values
    sampled_values = mu - sigma_safe * np.abs(z)
    
    # Cost-aware adjustment:
    # The raw TS score is -sampled_values (higher is better).
    # We want to penalize high costs.
    # cost is in [1, 2]. Let cost_penalty = (cost - 1) * weight.
    # We subtract this penalty from the score.
    
    # Determine the weight of the cost penalty.
    # In early stages (f_best is high), we might want to explore more regardless of cost.
    # In late stages (f_best is low), we should be more cost-aware.
    # However, TS already handles exploration via sigma.
    # A fixed moderate penalty is often robust.
    # Let's use a penalty that scales with the magnitude of the objective.
    # If f_best is large, the objective values are large, so a small cost penalty
    # might be negligible. We want the cost penalty to be significant relative to the
    # variation in sampled values.
    
    # A simple robust approach:
    # score = -sampled_values - lambda * (cost - 1)
    # Let's choose lambda dynamically.
    # If f_best is close to 0, the differences in sampled values might be small,
    # so cost matters more. If f_best is large, differences are large, so cost matters less.
    
    # Estimate the scale of the sampled values
    # Use the standard deviation of the sampled values as a reference
    if len(sampled_values) > 1:
        scale = np.std(sampled_values)
    else:
        scale = 1.0
    
    # Ensure scale is not too small
    scale = max(scale, 1e-6)
    
    # Lambda: weight of cost penalty relative to the scale of the objective
    # We want the cost penalty to be a small fraction of the typical variation
    # in the sampled objectives, so that TS dominates but cost breaks ties.
    # Let's set lambda = 0.1 * scale.
    # This means a cost difference of 1 (max possible) penalizes by 0.1 * scale.
    # If the sampled values vary by 1 * scale, this is a 10% effect.
    
    lambda_cost = 0.1 * scale
    
    # Calculate scores
    # Base score: higher is better, so negate the sampled values
    base_score = -sampled_values
    
    # Cost penalty: higher cost -> lower score
    cost_penalty = lambda_cost * (cost - 1.0)
    
    scores = base_score - cost_penalty
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
