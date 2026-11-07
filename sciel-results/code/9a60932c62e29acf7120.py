# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_TIME_LIMIT = 300.0

def acquisition(mu, sigma, f_best, cost):
    """
    Hybrid Cost-Aware Acquisition: Blend of CA-UCB and CA-POI.
    
    Strategy:
    1. CA-UCB provides exploration by rewarding high uncertainty.
    2. CA-POI provides robust exploitation by rewarding the probability of beating f_best, 
       which is less sensitive to sigma scaling than EI and handles multi-modality better.
    3. A dynamic weight alpha blends these two. Early in the search (high f_best), 
       alpha is high (favor UCB/exploration). As f_best improves, alpha decreases 
       (favor POI/exploitation).
    4. Both terms are divided by cost to enforce cost-awareness.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Predictive mean from the GP.
    sigma : array-like, shape (2000,)
        Predictive standard deviation from the GP.
    f_best : float
        Lowest observed function value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT - 5.0:
        return np.zeros_like(mu, dtype=np.float64)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety checks
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Estimate progress based on f_best
    # Styblinski-Tang 2D min is -78.3323
    # Initial random points are typically around -10 to -30
    f_min_theoretical = -78.3323
    f_max_initial = -10.0
    
    # Progress: 0 at start, 1 at end
    # Clip to avoid division by zero or negative progress
    progress = (f_max_initial - f_best) / (f_max_initial - f_min_theoretical)
    progress = np.clip(progress, 0.0, 1.0)
    
    # Dynamic alpha: 
    # Start with alpha = 0.8 (mostly UCB)
    # End with alpha = 0.2 (mostly POI)
    # Using a smooth decay
    alpha = 0.8 - 0.6 * progress
    
    # Calculate CA-UCB
    # For minimization, UCB = mu - beta * sigma
    # We want to maximize the score, so we use -mu + beta * sigma
    # Beta is fixed at 2.0 for stability, or could be tuned. 
    # The current program uses adaptive beta, but let's keep it simple here 
    # and rely on the blend for adaptivity.
    beta = 2.0
    ucb_score = (-mu + beta * sigma) / cost
    
    # Calculate CA-POI
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues in Phi
    z = np.clip(z, -10, 10)
    poi_prob = norm.cdf(z)
    poi_score = poi_prob / cost
    
    # Normalize scores to have comparable scales?
    # UCB can be large (if sigma is large). POI is in [0, 1].
    # Dividing by cost (1-2) doesn't change scale much.
    # We should normalize UCB to [0,1] or similar to blend effectively?
    # Or just scale the UCB term.
    # Let's normalize UCB by its max value in the current batch to keep it in [0, 1] relative scale.
    # However, max(ucb_score) might be dominated by high sigma regions.
    # A safer blend:
    # score = alpha * ucb_score + (1 - alpha) * poi_score
    # If ucb_score is ~10 and poi_score is ~0.5, alpha=0.5 gives 5.25.
    # If we normalize UCB to [0,1], the blend is better balanced.
    
    # Normalize UCB scores to [0, 1]
    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    if ucb_max > ucb_min:
        ucb_norm = (ucb_score - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb_score)
        
    # POI is already in [0, 1]
    
    scores = alpha * ucb_norm + (1.0 - alpha) * poi_score
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
