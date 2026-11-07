"""
Acquisition function for Bayesian optimization of Hartmann 3D.
Strategy: Thompson Sampling.
We sample a single function value from the posterior distribution N(mu, sigma^2)
for each candidate point and return these samples as scores.
This stochastic approach encourages exploration of regions with high uncertainty
without needing to tune a kappa parameter or explicitly penalize density,
often leading to better global convergence in multimodal functions.
"""

import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition scores.
    
    Args:
        mu: Array of GP posterior means (shape: 2000,)
        sigma: Array of GP posterior standard deviations (shape: 2000,)
        f_best: Best observed function value so far (scalar)
        
    Returns:
        Array of scores (shape: 2000,). Higher scores indicate candidates
        more likely to be selected for the next evaluation.
    """
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Deadline guard to ensure we return within time limits
    start_time = time.time()
    deadline = start_time + 290.0  # 290s limit, leaving 10s buffer for 300s total
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-12)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-12)
    
    # Sample from the posterior N(mu, sigma^2)
    # Since we only have mu and sigma, we can directly sample.
    # The evaluator will pick the candidate with the highest sample value.
    # For minimization, we want low function values.
    # Thompson Sampling typically samples f(x) ~ N(mu, sigma^2) and picks the one with the lowest sample
    # if we are minimizing. 
    # However, the interface says "higher meaning evaluate next".
    # So if we sample y ~ N(mu, sigma^2), a low y is good for minimization.
    # To make "higher score" mean "better for minimization", we should return -y.
    
    # Let's verify the logic:
    # If we are minimizing f(x), and we sample y from posterior,
    # a candidate with a very low y is promising.
    # We want the selection criterion to pick the best minimizer.
    # If the evaluator picks `argmax(scores)`, and we want it to pick the candidate
    # with the lowest expected/minimized value, we should set score = -sample.
    
    # Draw a single standard normal random number
    z = np.random.randn(n)
    
    # Compute samples
    samples = mu + sigma * z
    
    # Convert to maximization score: negate the samples
    scores = -samples
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Check time (unlikely to be an issue for simple sampling, but good practice)
    if time.time() > deadline:
         # Fallback to deterministic LCB if sampling took too long (should not happen)
         # But since it's just vectorized ops, this is extremely unlikely.
         pass
         
    return scores
