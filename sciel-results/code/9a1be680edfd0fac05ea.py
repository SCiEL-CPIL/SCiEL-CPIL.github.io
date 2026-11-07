"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Thompson Sampling acquisition: sample a single latent function value
    # for each candidate from the GP predictive distribution, then select
    # the candidate that maximizes the ratio of improvement to cost.
    # This stochastically balances exploration and exploitation.
    
    # Clamp sigma and cost for numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Sample from the GP predictive distribution: y = mu + sigma * z
    # where z ~ N(0, 1)
    z_samples = np.random.standard_normal(len(mu))
    sampled_values = mu + sigma * z_samples
    
    # For minimization, we want low values. The "improvement" of a sampled
    # point is how much better it is than f_best.
    # We use the negative of the sampled value as a proxy for "reward"
    # (since lower is better), or equivalently, we want to maximize
    # (f_best - sampled_value) / cost.
    
    # However, Thompson Sampling typically works by sampling and then
    # picking the argmin of the sampled objective (for minimization).
    # But we need to return scores for all 2000 candidates, and the
    # evaluator picks the one with the highest score.
    
    # A common approach for TS in BO is to sample the function and then
    # use the negative of the sampled value as the acquisition score.
    # But we also need to account for cost.
    
    # Let's use: score = (f_best - sampled_value) / cost
    # This encourages points that are likely to be much better than current
    # best, per unit cost.
    
    # To avoid division by zero or negative scores dominating, we can
    # also add a small constant to the denominator or use a different formulation.
    
    # Alternative: score = -sampled_value / cost
    # This directly minimizes the sampled objective per unit cost.
    
    # Let's try: score = (f_best - sampled_value) / (cost + 1e-9)
    # This gives positive scores for points likely to improve, and negative
    # for points likely to be worse.
    
    scores = (f_best - sampled_values) / cost
    
    # Add a small exploration bonus based on uncertainty to encourage
    # visiting unexplored regions.
    # The uncertainty is sigma. We can add a small multiple of sigma/cost.
    exploration_weight = 0.1
    scores = scores + exploration_weight * sigma / cost
    
    # Guarantee finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
