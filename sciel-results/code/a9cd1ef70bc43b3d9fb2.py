"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Multi-Seed Thompson Sampling.
    
    Args:
        mu: (2000,) array of GP means.
        sigma: (2000,) array of GP standard deviations.
        f_best: float, the lowest observed value so far.
        cost: (2000,) array of evaluation costs for each candidate.
        
    Returns:
        scores: (2000,) array of scores (higher is better).
    """
    start_time = time.time()
    deadline = start_time + 290.0  # 300s limit, keep 10s margin
    
    n_samples = 2000
    n_seeds = 10  # Number of Thompson Sampling draws
    
    # Pre-allocate scores to store the minimum value seen across seeds for each candidate
    # We want to maximize the "best case" among the random draws, but since we are minimizing f,
    # we actually want the sample value to be as low as possible.
    # Thompson Sampling for minimization: draw y ~ N(mu, sigma^2).
    # The score for a candidate is the drawn value y.
    # To make it robust, we take the minimum of the draws (most optimistic view) 
    # or the mean of the draws (expected value).
    # However, standard TS picks the argmin of the single draw.
    # To create a score vector for the evaluator to pick the best one, 
    # we can estimate the probability of being the minimum or use the mean of the draws.
    # Actually, the evaluator picks the candidate with the HIGHEST score.
    # If we use TS, we want to identify the candidate that is likely to have the lowest f(x).
    # So, for each candidate x_i, we draw y_{i,k} ~ N(mu_i, sigma_i^2).
    # A good score for x_i is the minimum of its draws min_k(y_{i,k}), 
    # but we need to invert this because higher score = better.
    # So score_i = -min_k(y_{i,k}).
    # This effectively prioritizes candidates that have at least one very low sample value.
    
    # However, we must divide by cost.
    # Score = -min_k(y_{i,k}) / cost_i
    
    # Optimization: 
    # Instead of storing all draws, we can accumulate the minimum.
    # y_{i,k} = mu_i + sigma_i * z_{i,k} where z ~ N(0,1)
    # min_k(mu_i + sigma_i * z_{i,k}) = mu_i + sigma_i * min_k(z_{i,k})
    # So we just need to find the minimum z across seeds for each candidate.
    # Wait, min(mu + sigma*z) is not mu + sigma*min(z) if sigma varies?
    # No, for a fixed i, mu_i and sigma_i are constant.
    # min_k (mu_i + sigma_i * z_{i,k}) = mu_i + sigma_i * min_k(z_{i,k})
    # This is correct.
    
    # So, we can generate an (n_seeds, n_samples) matrix of standard normal randoms.
    # Find the min along axis 0 -> min_z (n_samples,)
    # Score = -(mu + sigma * min_z) / cost
    
    # But wait, is min_z the best proxy?
    # If we take the min of the draws, we are taking the most optimistic sample.
    # This is equivalent to a quantile-based approach (approx 10th percentile if n_seeds=10).
    # The inspiration mentioned "Quantile-LCB (10th percentile)" improved score significantly.
    # Thompson Sampling with n_seeds=10 and taking the min approximates the 10th percentile of the predictive distribution.
    # This aligns with the "Quantile-Based" directive found in the cell findings.
    
    # Let's implement this efficiently.
    
    # 1. Generate random normals
    # Shape: (n_seeds, n_samples)
    # This is 10 * 2000 = 20,000 numbers. Very fast.
    rng = np.random.default_rng()
    
    min_z = np.zeros(n_samples)
    
    # We can do this in chunks if memory was an issue, but 20k floats is tiny.
    # However, to be safe with the deadline, we'll just do it in one go.
    
    # Generate all seeds at once
    z_samples = rng.standard_normal((n_seeds, n_samples))
    
    # Find the minimum z for each candidate across all seeds
    # This gives us the most "optimistic" standard normal deviation offset
    min_z = np.min(z_samples, axis=0)
    
    # Calculate the sampled values
    # y_min = mu + sigma * min_z
    sampled_values = mu + sigma * min_z
    
    # We want to minimize f, so lower sampled_values are better.
    # The acquisition score should be higher for better candidates.
    # So we negate the sampled value.
    raw_scores = -sampled_values
    
    # Apply cost penalty
    # The problem states: "Evaluating x costs 1 + mean(x scaled to [0, 1])"
    # But the `cost` parameter is passed in. We should use the provided `cost` vector.
    # Score = raw_scores / cost
    # Ensure no division by zero or negative costs (costs are positive by definition)
    # Add a small epsilon to cost to avoid division by zero if cost is 0 (unlikely but safe)
    scores = raw_scores / (cost + 1e-9)
    
    # Check deadline
    if time.time() > deadline:
         # If we are running out of time, return the current scores.
         # Since this computation is very fast (< 1ms usually), this is unlikely to trigger.
         pass
         
    return scores
# EVOLVE-BLOCK-END
