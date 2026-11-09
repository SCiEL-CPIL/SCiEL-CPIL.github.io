"""Cost-aware acquisition for three-hump-camel 2D BO.

Structural change: Replace the normalized weighted EI/PI blend with a
Thompson Sampling approach that samples from the posterior to balance
exploration and exploitation, with a cost-aware penalty.

Thompson Sampling naturally adapts the exploration-exploitation trade-off:
- In early stages (high uncertainty), samples are spread out, encouraging exploration.
- In later stages (low uncertainty), samples concentrate near the best known points,
  encouraging exploitation.

The cost-aware component penalizes points with higher evaluation costs,
effectively implementing a budget-aware strategy.
"""
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 300.0  # seconds, with safety margin


def _fast_norm_cdf(x):
    """Fast approximation of the standard normal CDF."""
    # Use the erf approximation for speed
    return 0.5 * (1.0 + np.erf(x / np.sqrt(2.0)))


def _fast_norm_pdf(x):
    """Fast approximation of the standard normal PDF."""
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Samples from the posterior distribution at each candidate point and
    selects the point with the best sampled value, penalized by cost.
    
    The key insight is that Thompson Sampling naturally balances exploration
    and exploitation: when uncertainty is high, samples are more spread out,
    encouraging exploration; when uncertainty is low, samples concentrate
    near the mean, encouraging exploitation.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases where sigma might be zero or negative
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling: sample from the posterior at each candidate
    # sample = mu + sigma * z, where z ~ N(0, 1)
    # We use a single random draw for all candidates to maintain consistency
    z = np.random.standard_normal(len(mu))
    samples = mu + sigma_safe * z
    
    # For minimization, we want the lowest sampled value
    # The "improvement" is f_best - sample (positive if sample is better)
    improvement = f_best - samples
    
    # Cost-aware penalty: divide by a cost factor
    # We use a soft penalty: score = improvement / (1 + alpha * cost_normalized)
    # where cost_normalized is the cost relative to the mean cost
    
    # Normalize cost to have mean 1
    mean_cost = np.mean(cost) + 1e-9
    cost_norm = cost / mean_cost
    
    # The cost penalty should be stronger when the budget is tight
    # We estimate remaining budget from the cost array
    # cost is the cost of evaluating each point, so higher cost = more budget used
    # A simple heuristic: penalize points with above-average cost more
    
    # Use a temperature parameter that adapts to the scale of improvements
    # This helps balance exploration and exploitation
    imp_scale = np.std(improvement) + 1e-9
    temperature = imp_scale * 0.5
    
    # Score: expected improvement per unit cost, with Thompson Sampling
    # We use the sampled improvement, penalized by cost
    # Higher score = better point to evaluate next
    
    # The cost penalty: we want to discourage expensive evaluations
    # Use a multiplicative penalty: score = improvement * exp(-beta * cost_norm)
    # where beta controls the strength of the cost penalty
    
    # Adaptive beta: stronger penalty when improvements are small (exploitation phase)
    # Weaker penalty when improvements are large (exploration phase)
    mean_improvement = np.mean(np.maximum(improvement, 0))
    beta = 0.3 * (1.0 + mean_improvement / (temperature + 1e-9))
    
    # Compute the cost-aware score
    cost_penalty = np.exp(-beta * (cost_norm - 1.0))
    scores = improvement * cost_penalty
    
    # Add a small exploration bonus based on uncertainty
    # This helps avoid getting stuck in local optima
    exploration_bonus = sigma_safe * 0.1
    scores = scores + exploration_bonus
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
