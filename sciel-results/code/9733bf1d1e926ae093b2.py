import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    Uses Probability of Improvement (PoI) which decouples exploration from
    improvement magnitude, encouraging broader coverage of the multi-modal
    Branin landscape to find more of the three global minima within 30 evaluations.
    
    For minimization, PoI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    This naturally balances exploration (high sigma) and exploitation (low mu)
    without the clustering behavior of EI.
    """
    start_time = time.time()
    
    # Handle zero or near-zero sigma to avoid division by zero
    eps = 1e-12
    sigma_safe = np.where(sigma > eps, sigma, eps)
    
    # PoI: Probability that f(x) < f_best
    # Z = (f_best - mu) / sigma, PoI = Phi(Z)
    Z = (f_best - mu) / sigma_safe
    scores = norm.cdf(Z)
    
    # Add a small UCB component to encourage exploration of high-uncertainty regions
    # This helps escape local optima in the multi-modal landscape
    # Use a modest beta to not overwhelm the PoI signal
    beta = 0.5
    ucb_component = beta * sigma_safe
    
    # Normalize UCB component to be comparable to PoI scores (which are in [0,1])
    ucb_max = np.max(ucb_component) + 1e-12
    ucb_normalized = ucb_component / ucb_max
    
    # Blend: primarily PoI with a small UCB boost for exploration
    w_poi = 0.85
    w_ucb = 0.15
    
    scores = w_poi * scores + w_ucb * ucb_normalized
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Safety margin check
    elapsed = time.time() - start_time
    if elapsed > 280.0:
        # Return a simple fallback if we're running out of time
        return norm.cdf((f_best - mu) / sigma_safe)
    
    return scores
