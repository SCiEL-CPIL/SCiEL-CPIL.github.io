"""Cost-aware acquisition with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 325.0


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback: Cost-aware mean
        return mu / cost

    # Upper Confidence Bound (UCB) per unit cost
    # Beta controls exploration. 1.0 is a moderate choice.
    beta = 1.0
    scores = (mu - beta * sigma) / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
