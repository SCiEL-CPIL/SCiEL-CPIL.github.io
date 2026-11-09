import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Upper Confidence Bound (UCB) acquisition function for minimization.
    
    Args:
        mu: (2000,) Mean of the GP predictions.
        sigma: (2000,) Standard deviation of the GP predictions.
        f_best: float, the best (lowest) value observed so far.
        cost: (2000,) Cost of evaluating each candidate.
        
    Returns:
        scores: (2000,) Acquisition scores. Higher scores indicate better candidates.
    """
    t_start = time.time()
    
    # Convert inputs to numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety measures
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Dynamic Kappa Heuristic:
    # Since we don't have the global step count, we estimate the exploration phase
    # by looking at the average uncertainty. 
    # High average sigma suggests early exploration or unexplored regions.
    # Low average sigma suggests late exploitation.
    # We use a base kappa and modulate it slightly.
    # For Hartmann 3D, a kappa of 2.0 is a robust starting point.
    # If sigma is very high, we might want to be even more exploratory? 
    # Or is the linear scaling of kappa*sigma sufficient?
    # Let's stick to a fixed kappa to avoid instability from noisy sigma estimates.
    kappa = 2.0
    
    # UCB for minimization:
    # We want to minimize the function. The optimistic estimate of the minimum value
    # at a point is mu - kappa * sigma.
    # We want to select points with the lowest optimistic value, penalized by cost.
    # To convert "minimize (mu - kappa*sigma)/cost" to "maximize score", we negate it:
    # score = -(mu - kappa * sigma) / cost = (kappa * sigma - mu) / cost
    
    scores = (kappa * sigma_safe - mu) / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard (should be instantaneous for N=2000)
    if time.time() - t_start > 290:
        pass
        
    return scores
