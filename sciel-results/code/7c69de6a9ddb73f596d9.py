"""Rewritten program using a novel approach: Cost-aware Upper Confidence Bound with adaptive exploration/exploitation balance."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of Expected Improvement and
    Upper Confidence Bound, with adaptive scaling based on cost structure.
    
    The key insight: instead of simple EI/cost, we use a modified UCBC that
    balances exploration (sigma) and exploitation (f_best - mu) more effectively,
    and applies a non-linear cost penalty that accounts for the fact that cheaper
    points (lower x) should be preferred when the improvement is marginal.
    """
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Upper Confidence Bound (UCB) component
    # Use a beta that adapts based on the scale of sigma
    # When sigma is large (early on), we want more exploration
    # When sigma is small (late), we want more exploitation
    beta = 2.0  # exploration-exploitation balance
    
    ucb = -(mu + beta * sigma_safe)  # negative because we minimize
    
    # Cost-aware weighting:
    # The cost is 1 + mean(x_scaled), so cost ranges from 1 to 2.
    # We use a power-law cost penalty that is more aggressive for expensive points
    # when the expected improvement is not substantial.
    
    # Normalize EI to get a sense of "value"
    # Use a soft threshold: if EI is very small relative to sigma, penalize cost more
    ei_scaled = ei / (sigma_safe + 1e-9)
    
    # Cost penalty: higher cost -> lower score, but with diminishing returns
    # cost is in [1, 2], so 1/cost is in [0.5, 1]
    # We use cost^(-gamma) where gamma adapts to the situation
    # When EI is high relative to uncertainty, we can tolerate higher cost
    gamma = 1.0 + 0.5 * np.exp(-ei_scaled)  # gamma in [1, 1.5]
    
    cost_penalty = cost ** (-gamma)
    
    # Combine EI and UCB with a weighted sum
    # EI is in objective units, UCB is also in objective units (negated for minimization)
    # We need to balance them properly
    
    # Scale EI and UCB to comparable ranges
    # EI is typically positive when mu < f_best
    # UCB is negative (we negate it)
    
    # Use a convex combination
    alpha = 0.6  # weight for EI
    score = alpha * ei + (1 - alpha) * (-mu - beta * sigma_safe)
    
    # Apply cost penalty
    score = score * cost_penalty
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
