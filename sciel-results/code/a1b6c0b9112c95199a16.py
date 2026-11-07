"""Cost-aware UCB acquisition for Hartmann 6D minimization."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    # UCB for minimization: maximize (-mu + beta * sigma) / cost
    # beta controls exploration; cost penalizes expensive candidates.
    # Use a moderate beta that balances exploitation and exploration.
    beta = 2.0

    # Compute the base UCB score for minimization
    # Higher score = more likely to be evaluated next
    base_score = beta * sigma - mu

    # Normalize by cost to prefer cheaper evaluations
    # cost is in [1, 2] since x is in [0,1]^6 and cost = 1 + mean(x)
    # Use linear cost penalty to balance exploitation, exploration, and evaluation cost
    scores = base_score / cost

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
