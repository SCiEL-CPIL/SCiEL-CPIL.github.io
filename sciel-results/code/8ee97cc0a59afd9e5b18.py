"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Thompson Sampling: draw a sample from the posterior at each candidate,
    # then prefer the candidate whose sampled objective is lowest per unit cost.
    # This balances exploration/exploitation without EI's bias toward good regions.
    #
    # We use a deterministic pseudo-random draw that is a function of the candidate
    # index (via a hash of the position) so that the selection is stable across calls
    # within a single evaluation round, but effectively random across rounds because
    # the mu/sigma arrays change. Actually, to be safe and simple, we use a single
    # random draw per candidate based on a global counter or just np.random.
    #
    # However, the task says "score 2000 new uniform candidates with acquisition and
    # evaluates the best one". The acquisition function is called once per round with
    # all 2000 candidates. We need to return a score for each.
    #
    # For Thompson Sampling, we can sample y_i ~ N(mu_i, sigma_i^2) for each candidate i,
    # and then the "best" is argmin y_i. But we need to return a score such that higher
    # = evaluate next. So we can return -y_i / cost_i, or just -y_i if cost is constant,
    # but cost varies.
    #
    # To maintain cost-awareness, we divide the sampled value by cost.
    # But we need a single random draw per candidate to make it a proper TS.
    #
    # Use np.random.standard_normal to generate the draws.
    n = len(mu)
    # Generate random samples from the posterior
    samples = mu + sigma * np.random.standard_normal(n)
    # Higher score = better (lower objective, lower cost)
    # We want to minimize samples / cost, so score = -samples / cost
    # But to avoid division by zero and ensure finite scores:
    cost_safe = np.maximum(cost, 1e-10)
    scores = -samples / cost_safe
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
# EVOLVE-BLOCK-END
