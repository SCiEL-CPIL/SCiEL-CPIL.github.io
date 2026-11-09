import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Compute a blended acquisition function combining Expected Improvement (EI)
    and Upper Confidence Bound (UCB).
    
    Parameters:
    mu : np.ndarray
        Posterior mean at candidate points, shape (N,).
    sigma : np.ndarray
        Posterior standard deviation at candidate points, shape (N,).
    f_best : float
        Best observed value so far (minimum).
    
    Returns:
    scores : np.ndarray
        Blended scores, shape (N,), higher is better.
    """
    # Handle numerical stability for sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Expected Improvement (EI)
    # Improvement is f_best - f(x). Since we minimize, lower f(x) is better.
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Upper Confidence Bound (UCB)
    # UCB = -mu + beta * sigma
    # We use a beta of 1.0 as a reasonable default for exploration.
    beta = 1.0
    ucb = -mu + beta * sigma_safe
    
    # 3. Normalize scores to comparable scales
    # This prevents one term from dominating due to magnitude differences.
    def normalize(x):
        mean = np.mean(x)
        std = np.std(x)
        if std < 1e-9:
            return np.zeros_like(x)
        return (x - mean) / std
    
    ei_norm = normalize(ei)
    ucb_norm = normalize(ucb)
    
    # 4. Blend
    # Based on the knowledge provided, 70/30 weighting worked well.
    w_ei = 0.7
    w_ucb = 0.3
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
