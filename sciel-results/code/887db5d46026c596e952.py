# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: expected improvement per unit cost, with
    additional exploration bonus based on uncertainty and a penalty for
    high cost. Uses a combination of EI/cost and a UCB-like term.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalize EI by cost (cost-aware)
    ei_cost = ei / (cost + 1e-9)
    
    # UCB-style exploration term: balances exploitation and exploration
    # The beta parameter controls exploration vs exploitation
    beta = 0.5
    ucb = mu - beta * sigma_safe  # For minimization, we want low mu and high sigma
    
    # Combine: primary is EI per cost, secondary is exploration
    # We want to maximize this score
    # Normalize ucb to be comparable to ei_cost
    # Since mu and sigma are in objective units, and ei_cost is in objective/cost units,
    # we need a way to combine them. Let's use a weighted sum with scaling.
    
    # Actually, let's think about this differently.
    # The task says "higher = evaluate next", so we want to maximize the return.
    # EI/cost is a good baseline. Let's add a small exploration bonus.
    
    # A simpler and more robust approach: use EI/cost but with a floor on cost
    # to avoid division by very small costs, and add a small sigma-based bonus
    # to encourage exploration in uncertain regions.
    
    # Let's try: score = EI / cost + lambda * sigma / cost
    # This encourages both expected improvement and exploration, weighted by cost.
    
    lambda_explore = 0.1
    score = ei_cost + lambda_explore * sigma_safe / (cost + 1e-9)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
