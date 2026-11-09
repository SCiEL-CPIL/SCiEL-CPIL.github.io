import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a Structural Shift: 
    From deterministic ranking (UCB/EI) to Stochastic Multi-Armed Bandit (Thompson Sampling).
    
    Strategy:
    1. Instead of computing a single score for each point (deterministic ranking), we simulate
       multiple "worlds" (posterior samples) of the function values at the 2000 candidate points.
    2. In each world, we identify the point with the lowest predicted value (the "winner" of that world).
    3. We count how often each point wins across all worlds. This frequency is the probability
       that the point is the global minimum in the current posterior distribution.
    4. We select the point that maximizes the probability of being the minimum, weighted by 
       cost-efficiency (1/cost).
    5. This approach naturally balances exploration and exploitation: 
       - High uncertainty (sigma) leads to high variance in samples, giving a point a chance to win 
         even if its mean is not the best.
       - Low uncertainty leads to concentrated samples, so only points with low means win.
    
    This is fundamentally different from UCB/EI because it does not rely on a specific confidence 
    bound or improvement expectation formula, but on the empirical distribution of the argmin.
    """
    
    start_time = time.time()
    deadline = start_time + 290.0  # 300s limit, keep 10s margin
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-6)
    
    # Normalize cost to [0, 1] range for better scaling? 
    # Cost is 1 + mean(x_scaled). x_scaled in [0,1]. So cost in [1, 2].
    # We want to prefer lower cost. Score ~ 1/cost.
    
    n_points = len(mu)
    n_samples = 50  # Number of posterior samples to simulate
    
    # Pre-allocate counts
    win_counts = np.zeros(n_points, dtype=np.float64)
    
    # Simulate posterior samples
    # Since we don't have the full covariance matrix, we assume independence for the marginal samples.
    # This is an approximation of Thompson Sampling.
    for _ in range(n_samples):
        # Sample from N(mu, sigma)
        f_samples = mu + sigma * np.random.randn(n_points)
        
        # Find the index with the minimum value in this sample
        min_idx = np.argmin(f_samples)
        win_counts[min_idx] += 1.0
    
    # Probability of being the minimum
    prob_min = win_counts / n_samples
    
    # Score: Probability of being minimum, penalized by cost
    # We want to maximize this score.
    # score = prob_min / cost
    scores = prob_min / cost
    
    # Ensure finite scores
    scores[~np.isfinite(scores)] = 0.0
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        # If we ran out of time (unlikely for 50 samples), return current scores
        pass
    
    return scores
