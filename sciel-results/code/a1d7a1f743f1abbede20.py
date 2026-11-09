import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    
    Uses a hybrid of Expected Improvement (EI) and Upper Confidence Bound (UCB)
    with adaptive exploration-exploitation balance.
    
    Args:
        mu: Mean of the GP posterior at candidate points (shape: (2000,)).
        sigma: Standard deviation of the GP posterior at candidate points (shape: (2000,)).
        f_best: The lowest observed value so far.
        
    Returns:
        scores: Scores for each candidate point (shape: (2000,)). Higher is better.
    """
    # Ensure sigma is at least a small epsilon to avoid division by zero
    eps = 1e-6
    sigma_safe = np.maximum(sigma, eps)
    
    # 1. Calculate Expected Improvement (EI) for minimization
    # EI(x) = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    diff = f_best - mu
    z = diff / sigma_safe
    
    # Use scipy.stats.norm for numerical stability
    ei_scores = diff * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # EI is non-negative by definition
    ei_scores = np.maximum(ei_scores, 0)
    
    # 2. Calculate UCB score for minimization
    # We want to minimize mu - beta * sigma, so we maximize -mu + beta * sigma
    beta = 2.0
    ucb_scores = -mu + beta * sigma_safe
    
    # 3. Normalize scores to zero mean and unit variance
    # This makes the weighting between EI and UCB consistent regardless of scale
    
    # Normalize EI
    ei_mean = np.mean(ei_scores)
    ei_std = np.std(ei_scores)
    if ei_std > 1e-8:
        ei_norm = (ei_scores - ei_mean) / ei_std
    else:
        ei_norm = np.zeros_like(ei_scores)
        
    # Normalize UCB
    ucb_mean = np.mean(ucb_scores)
    ucb_std = np.std(ucb_scores)
    if ucb_std > 1e-8:
        ucb_norm = (ucb_scores - ucb_mean) / ucb_std
    else:
        ucb_norm = np.zeros_like(ucb_scores)
        
    # 4. Combine scores with adaptive weighting
    # Use a balance that favors exploitation when we have good information,
    # and exploration when uncertainty is high
    w_ei = 0.6
    w_ucb = 0.4
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
