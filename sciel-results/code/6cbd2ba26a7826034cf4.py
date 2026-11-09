import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) and
    Upper Confidence Bound (UCB) for minimization.
    
    Args:
        mu: GP posterior mean, shape (N,)
        sigma: GP posterior std, shape (N,)
        f_best: Best observed value so far (lower is better)
        
    Returns:
        Scores for each candidate, shape (N,). Higher scores preferred.
    """
    # Ensure sigma is strictly positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement for minimization
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB term for exploration: -mu + beta * sigma
    # beta=1.0 is a moderate exploration coefficient
    beta = 1.0
    ucb = -mu + beta * sigma_safe
    
    # Normalize both to comparable scales [0, 1]
    # This prevents one term from dominating solely due to scale differences
    def _normalize(arr):
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        rng = arr_max - arr_min
        if rng < 1e-12:
            return np.zeros_like(arr)
        return (arr - arr_min) / rng
    
    ei_norm = _normalize(ei)
    ucb_norm = _normalize(ucb)
    
    # Weighted combination: emphasize EI but allow UCB to influence
    # This helps escape local optima while still exploiting known good regions
    w_ei = 0.7
    w_ucb = 0.3
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
