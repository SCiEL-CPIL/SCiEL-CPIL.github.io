import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    Uses Probability of Improvement (PoI) as the primary signal, which decouples
    exploration from improvement magnitude, encouraging broader coverage of the
    multi-modal Branin landscape. A small normalized sigma term provides mild
    exploration diversity without overwhelming the PoI signal.
    
    For minimization, PoI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    """
    start_time = time.time()
    
    # Handle zero or near-zero sigma to avoid division by zero
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # PoI: Probability that f(x) < f_best
    # Z = (f_best - mu) / sigma, PoI = Phi(Z)
    Z = (f_best - mu) / sigma_safe
    poi = norm.cdf(Z)
    
    # Add a small exploration term based on normalized sigma
    # This helps in multi-modal landscapes like Branin by providing
    # mild diversity without dominating the PoI signal
    exploration_weight = 0.05
    sigma_norm = sigma / (np.max(sigma) + eps)
    exploration = exploration_weight * sigma_norm
    
    # Combine: PoI dominates, exploration provides mild diversity
    scores = poi + exploration
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Safety margin check
    elapsed = time.time() - start_time
    if elapsed > 280.0:
        # Return a simple fallback if we're running out of time
        return norm.cdf((f_best - mu) / sigma_safe)
    
    return scores
