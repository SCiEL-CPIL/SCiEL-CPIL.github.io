import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    
    Strategy: Predictive Entropy Search (PES) variant approximated by a combination of
    Probability of Improvement (PI) and Expected Improvement (EI), weighted to balance
    exploration and exploitation for the multimodal Weierstrass landscape.
    
    This targets the global structure by prioritizing points that are likely to improve
    upon the current best (PI) and the magnitude of that improvement (EI), which together
    approximate the expected reduction in posterior entropy of the global minimum.
    """
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit
    
    # Ensure numerical stability for sigma
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement (PI): P(f(x) < f_best)
    pi = norm.cdf(z)
    
    # Expected Improvement (EI) for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Check time
    if time.time() > deadline - 5.0:
        # Fallback to simple UCB if running out of time
        beta = 1.5
        return -mu + beta * sigma_safe
    
    # Adaptive weighting between PI and EI
    # When f_best is high (early), weight PI more (explore)
    # When f_best is low (late), weight EI more (exploit)
    f_best_clipped = np.clip(f_best, 0.0, 10.0)
    weight_pi = 0.3 + 0.7 * np.exp(-f_best_clipped)
    weight_ei = 1.0 - weight_pi
    
    # Normalize EI to be comparable to PI (which is in [0, 1])
    # EI can be larger than 1, so scale it
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ei_normalized = ei / ei_max
    
    # Combine PI and normalized EI
    scores = weight_pi * pi + weight_ei * ei_normalized
    
    # Add a small exploration bonus for high sigma regions
    # This helps escape local minima in the multimodal Weierstrass landscape
    sigma_max = np.max(sigma_safe) if np.max(sigma_safe) > 0 else 1.0
    exploration_bonus = 0.1 * (sigma_safe / sigma_max)
    
    scores = scores + exploration_bonus
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
