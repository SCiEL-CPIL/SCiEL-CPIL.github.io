"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = 330 - 5  # Safety margin

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() > _DEADLINE:
        # Fallback to a simple greedy strategy if we are running out of time
        return -mu / (cost + 1e-9)
    
    # Ensure sigma is non-negative and handle numerical stability
    sigma = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling:
    # Instead of calculating Expected Improvement (EI) which is deterministic 
    # and can get stuck in local optima or over-exploit known regions, 
    # we sample from the posterior distribution of the function values.
    
    # We want to maximize the "value" of an evaluation. 
    # The value is roughly the improvement over f_best, but penalized by cost.
    # In Thompson Sampling, we draw a sample y ~ N(mu, sigma^2).
    # We then score candidates based on how good that sample is, adjusted for cost.
    
    # A common cost-aware formulation for minimization is:
    # Score = (f_best - y_sample) / cost
    # However, since y_sample is random, we can use the expected value of this ratio?
    # No, Thompson Sampling works by sampling y, then picking the best x for that y.
    # But here, we must return a score for each x that the external loop will use to pick the best x.
    # So, if we sample y_x ~ N(mu_x, sigma_x^2), and return (f_best - y_x) / cost_x,
    # the external loop will pick the x with the highest score.
    # This is effectively one iteration of Thompson Sampling where the "sample" is drawn once for all candidates.
    
    # To make it more robust, we can draw multiple samples and average, or just use one.
    # One sample is standard for Thompson Sampling in this context.
    
    # Draw samples
    # Using a single draw per call is efficient.
    samples = mu + sigma * np.random.randn(len(mu))
    
    # Calculate improvement
    improvement = f_best - samples
    
    # Cost-aware score
    # We divide by cost to prefer cheaper points.
    scores = improvement / (cost + 1e-9)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
