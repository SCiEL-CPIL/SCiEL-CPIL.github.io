"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    # Start time for deadline guard
    start_time = time.time()
    deadline = 320.0  # 10s margin before 330s kill
    
    # Ensure cost is an array
    cost = np.asarray(cost)
    
    # Thompson Sampling: Sample from the GP posterior
    # mu and sigma are in objective units
    # We sample y ~ N(mu, sigma^2)
    # To avoid numerical issues with sigma=0, we add a small epsilon
    eps = 1e-9
    sigma_safe = sigma + eps
    
    # Generate random samples
    z = np.random.normal(0, 1, size=mu.shape)
    y_samples = mu + sigma_safe * z
    
    # Cost-aware scoring
    # We want to minimize y, but maximize the acquisition score.
    # Cost c = 1 + mean(x). Lower cost is better.
    # Score = -y / c. 
    # If y is very low (good), -y is high (good).
    # If c is low (cheap), -y/c is higher (good).
    
    # Check for non-finite values or zero costs (though cost >= 1)
    valid_mask = np.isfinite(y_samples) & (cost > eps)
    
    scores = np.zeros_like(y_samples)
    # Calculate scores for valid entries
    scores[valid_mask] = -y_samples[valid_mask] / cost[valid_mask]
    
    # For invalid entries (shouldn't happen if mu, sigma are finite), set to -inf or 0
    # The prompt requires finite scores. 
    # If mu or sigma are nan/inf, the result is invalid.
    # We should sanitize the output to be finite.
    
    # If any scores are not finite, replace them with the minimum finite score or 0
    if not np.all(np.isfinite(scores)):
        finite_scores = scores[np.isfinite(scores)]
        if len(finite_scores) > 0:
            min_score = np.min(finite_scores)
            scores[~np.isfinite(scores)] = min_score - 1.0
        else:
            scores = np.zeros_like(scores)
            
    # Deadline check
    if time.time() - start_time > deadline:
        # Fallback: If somehow too slow (unlikely for TS), return EI-like score
        # But TS is O(N), so this is a safety net.
        pass
        
    return scores
# EVOLVE-BLOCK-END
