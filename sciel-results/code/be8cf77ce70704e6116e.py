import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 320s to be safe within 330s

def acquisition(mu, sigma, f_best):
    """
    Compute a hybrid acquisition function combining Expected Improvement (EI)
    and a knowledge-gradient-like term, with adaptive exploration.
    
    Parameters:
    mu : np.ndarray
        Posterior mean at candidate points, shape (N,).
    sigma : np.ndarray
        Posterior standard deviation at candidate points, shape (N,).
    f_best : float
        Best observed value so far (minimum).
    
    Returns:
    scores : np.ndarray
        Scores, shape (N,), higher is better.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a simple UCB to be safe
        sigma_safe = np.maximum(sigma, 1e-9)
        return -mu + 2.0 * sigma_safe
    
    # Handle numerical stability for sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Expected Improvement (EI)
    # For minimization: EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    # where z = (f_best - mu(x)) / sigma(x)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Knowledge Gradient (KG) approximation
    # KG approximates the expected improvement in f_best after evaluating x
    # A common approximation: KG ~ EI(x) + sigma(x) * (some exploration term)
    # We use a simpler form: the probability that x improves f_best, weighted by potential improvement
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    prob_improve = norm.cdf(z)
    # Expected improvement magnitude
    exp_improve = np.maximum(f_best - mu, 0)
    kg = prob_improve * exp_improve + sigma_safe * norm.pdf(z)
    
    # 3. Upper Confidence Bound (UCB) for exploration
    # Use an adaptive beta based on the spread of sigma
    beta = 1.5 + 0.5 * np.std(sigma_safe) / (np.mean(sigma_safe) + 1e-9)
    ucb = -mu + beta * sigma_safe
    
    # 4. Normalize all three scores to comparable scales
    def normalize(x):
        mean = np.mean(x)
        std = np.std(x)
        if std < 1e-9:
            return np.zeros_like(x)
        return (x - mean) / std
    
    ei_norm = normalize(ei)
    kg_norm = normalize(kg)
    ucb_norm = normalize(ucb)
    
    # 5. Blend with adaptive weights
    # More exploration early (high sigma), more exploitation later (low sigma)
    sigma_mean = np.mean(sigma_safe)
    sigma_max = np.max(sigma_safe)
    # Exploration ratio: higher when sigma is large relative to f_best scale
    f_scale = abs(f_best) + 1e-9
    exploration_ratio = min(1.0, sigma_mean / f_scale)
    
    w_ei = 0.5 * (1 - exploration_ratio) + 0.3
    w_kg = 0.3 * (1 - exploration_ratio) + 0.2
    w_ucb = 0.2 * exploration_ratio + 0.1
    
    # Normalize weights to sum to 1
    w_sum = w_ei + w_kg + w_ucb
    w_ei /= w_sum
    w_kg /= w_sum
    w_ucb /= w_sum
    
    scores = w_ei * ei_norm + w_kg * kg_norm + w_ucb * ucb_norm
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
