import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on powell 4d.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Predicted mean of the GP at the 2000 candidate points
    sigma : array of shape (2000,)
        Predicted standard deviation of the GP at the 2000 candidate points
    f_best : float
        Best (lowest) observed function value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate: 1 + mean(x_scaled)
    
    Returns:
    --------
    score : array of shape (2000,)
        Acquisition scores (higher = prefer to evaluate next)
    """
    start_time = time.time()
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Structural Change: Use Upper Confidence Bound (UCB) instead of Expected Improvement (EI)
    # UCB for minimization: Potential Improvement + Exploration Bonus
    # Improvement part: f_best - mu (clamped to 0 to avoid penalizing regions with worse mean too heavily, 
    # though UCB usually doesn't clamp, clamping makes it more similar to "Optimistic EI")
    
    improvement = np.maximum(f_best - mu, 0.0)
    
    # Adaptive Beta:
    # We estimate the stage of the search by the magnitude of uncertainty.
    # High uncertainty -> Early stage -> Explore more (higher beta).
    # Low uncertainty -> Late stage -> Exploit more (lower beta).
    
    mean_sigma = np.mean(sigma_safe)
    max_sigma = np.max(sigma_safe)
    
    # Normalize uncertainty to [0, 1]
    # If max_sigma is 0, we are done, but let's handle it safely.
    if max_sigma > 1e-10:
        # Ratio of mean to max sigma. 
        # If sigma is concentrated (high ratio), we are confident.
        # If sigma is spread (low ratio), we are uncertain.
        # Let's use mean_sigma directly scaled by a characteristic scale of the problem.
        # The function value range of Powell is not fixed, but let's assume sigma decays.
        # A simple heuristic: beta is proportional to mean_sigma.
        # We need to map mean_sigma to a reasonable beta range, e.g., [0.5, 2.0].
        
        # Estimate the "scale" of the problem. 
        # If we don't know the scale, we can use the current f_best magnitude? No, f_best is value.
        # Let's use a fixed scaling factor based on the fact that sigma is in objective units.
        # A typical sigma might be O(1) to O(10).
        
        # Let's try: beta = 1.0 + 2.0 * (mean_sigma / (mean_sigma + 1.0))
        # This saturates at 3.0 for large sigma and approaches 1.0 for small sigma.
        # Or simpler: beta = 1.5 * (mean_sigma / (mean_sigma + 0.5)) + 0.5
        # Let's stick to a robust adaptive beta.
        
        # Use the ratio of mean to max to gauge "confidence".
        # High confidence (low ratio) -> lower beta.
        # Low confidence (high ratio) -> higher beta.
        sigma_ratio = mean_sigma / max_sigma
        
        # Map ratio [0, 1] to beta [2.0, 0.5]
        # ratio 0 (all uncertainty at one point, rare) -> high beta?
        # ratio 1 (uniform uncertainty) -> low beta?
        # Actually, if uncertainty is uniform and high, we should explore.
        # If uncertainty is low everywhere, we should exploit.
        
        # Let's use mean_sigma as the primary driver.
        # Assume a characteristic scale of 1.0 for the function values.
        # If mean_sigma is 5, beta should be high.
        # If mean_sigma is 0.1, beta should be low.
        
        # beta = 1.0 + 1.0 * np.tanh(mean_sigma)
        # tanh(0)=0, tanh(1)=0.76, tanh(5)=1.
        # So beta ranges from 1.0 to 2.0.
        
        beta = 1.0 + 1.5 * np.tanh(mean_sigma)
    else:
        beta = 0.5

    # UCB Score: (Improvement + Beta * Sigma) / Cost
    # This is a cost-weighted UCB.
    ucb_score = (improvement + beta * sigma_safe) / cost
    
    # Ensure all scores are finite
    score = np.nan_to_num(ucb_score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check time budget - safety margin before 300s limit
    elapsed = time.time() - start_time
    if elapsed > 250.0:
        # Return a simpler score to avoid timeout
        # In case of timeout, just return EI/cost as fallback? 
        # But we are fast. This is just a safety net.
        simple_score = improvement / cost
        simple_score = np.nan_to_num(simple_score, nan=0.0, posinf=0.0, neginf=0.0)
        return simple_score
    
    return score
