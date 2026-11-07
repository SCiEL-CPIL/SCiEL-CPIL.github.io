"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're close to the 300s limit, return a safe fallback
    # that still produces finite scores (e.g., expected improvement)
    elapsed = time.time() - _ACQ_START_TIME
    if elapsed > 295:  # safety margin of 5s
        # Fallback: simple EI
        z = (f_best - mu) / (sigma + 1e-9)
        return ((f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)) / cost
    
    # Cost-aware acquisition: balance expected improvement with cost
    # Use a combination of EI and UCB, weighted by cost efficiency
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB component: explore where uncertainty is high
    beta = 0.5
    ucb = mu + beta * sigma
    
    # Cost-aware: prefer points with high EI per unit cost
    # Also add a small exploration bonus for low-cost regions
    cost_efficiency = ei / (cost + 1e-9)
    exploration_bonus = -0.1 * (cost - 1.0)  # slight preference for cheaper points
    
    return cost_efficiency + exploration_bonus


# Track start time for deadline guard
_ACQ_START_TIME = time.time()
# EVOLVE-BLOCK-END
