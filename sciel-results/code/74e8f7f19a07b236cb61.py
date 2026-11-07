import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient (KG) acquisition function.
    Estimates the expected reduction in the global minimum for each candidate.
    """
    N = len(mu)
    
    # Set up a deadline guard to ensure we finish within time limits
    start_time = time.time()
    deadline = start_time + 5.0  # 5 seconds should be plenty for 2000 candidates
    
    # Number of Monte Carlo samples for KG estimation
    # We need enough samples for accurate KG, but not too many for speed
    K = 50
    
    # Generate samples from the posterior
    # Z shape: (K, N)
    Z = np.random.randn(K, N)
    
    # Y shape: (K, N)
    # Y[k, i] is the sampled value at candidate i for sample k
    Y = mu[None, :] + sigma[None, :] * Z
    
    # For each sample k, compute the current best (minimum) value
    # and then for each candidate i, compute what the new best would be
    # if we evaluated at i
    
    # Current best for each sample: shape (K,)
    current_best = np.min(Y, axis=1)
    
    # KG for each candidate i:
    # For each sample k, the improvement if we evaluate at i is:
    # max(0, current_best[k] - Y[k, i])
    # But we need to account for the fact that after evaluating at i,
    # the new best is min(current_best[k], Y[k, i])
    # The KG is E[current_best - new_best] = E[max(0, current_best - Y[i])]
    
    # Compute the improvement for each (k, i) pair
    # improvement[k, i] = max(0, current_best[k] - Y[k, i])
    # Shape: (K, N)
    improvement = np.maximum(0, current_best[:, None] - Y)
    
    # Average over samples to get KG for each candidate
    # Shape: (N,)
    scores = np.mean(improvement, axis=0)
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
