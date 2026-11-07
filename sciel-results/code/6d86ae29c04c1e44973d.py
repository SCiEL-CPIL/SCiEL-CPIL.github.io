# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    # Thompson Sampling: draw a single sample from the GP posterior
    # and select the candidate that minimizes (sample / cost).
    # Directive: Use sample = mu - sigma * z
    
    # Draw a single random standard normal sample
    z = np.random.standard_normal()
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # Draw the sample: mu - sigma * z
    # This represents a sample from the posterior distribution N(mu, sigma^2)
    sample = mu - sigma * z
    
    # Compute score: higher is better.
    # We want to minimize sample / cost.
    # So we return the negative of that ratio.
    cost_safe = np.maximum(cost, 1e-9)
    
    score = -sample / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
