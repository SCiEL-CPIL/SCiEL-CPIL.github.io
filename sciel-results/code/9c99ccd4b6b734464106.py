import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Ackley 2D.
    Uses a hybrid of UCB and Expected Improvement (EI) with adaptive weights.
    
    Parameters:
    -----------
    mu : numpy.ndarray, shape (2000,)
        GP mean predictions
    sigma : numpy.ndarray, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far (minimum)
    
    Returns:
    --------
    score : numpy.ndarray, shape (2000,)
        Acquisition scores (higher is better)
    """
    # Safety margin: ensure we don't exceed time budget
    t0 = time.time()
    
    # Standardize sigma for stable kappa scaling
    sigma_std = np.std(sigma)
    if sigma_std < 1e-12:
        sigma_std = 1.0
    
    # Adaptive UCB kappa: balance exploration/exploitation
    # When f_best is far from optimum (0), explore more
    # Use a smooth transition
    # f_best ranges from ~0 to ~10+ for Ackley
    # kappa_base = 1.0 was the proven baseline
    # Scale slightly based on uncertainty level
    
    # Normalize f_best: assume range [0, 10]
    f_norm = np.clip(f_best / 10.0, 0.0, 1.0)
    
    # kappa decreases as we get closer to optimum
    # At f_best=0: kappa = 0.5 (exploit)
    # At f_best=10: kappa = 2.0 (explore)
    kappa = 0.5 + 1.5 * f_norm
    
    # UCB score for minimization: -mu + kappa * sigma
    ucb_score = -mu + kappa * sigma
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # But we want to maximize EI, so we compute it directly
    from scipy.stats import norm
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    
    # EI formula for minimization
    ei_score = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei_score = np.maximum(ei_score, 0.0)
    
    # Hybrid: combine UCB and EI
    # Weight: more UCB when uncertain, more EI when confident
    # Use mean sigma to determine confidence
    mean_sigma = np.mean(sigma)
    
    # If mean_sigma is high, we're uncertain -> weight UCB more
    # If mean_sigma is low, we're confident -> weight EI more
    # Normalize mean_sigma: assume range [0, 5]
    sigma_norm = np.clip(mean_sigma / 5.0, 0.0, 1.0)
    
    # Weight for UCB: 0.3 to 0.7
    w_ucb = 0.3 + 0.4 * sigma_norm
    w_ei = 1.0 - w_ucb
    
    # Normalize both scores to [0, 1] range for fair combination
    def normalize(s):
        s_min = np.min(s)
        s_max = np.max(s)
        if s_max - s_min < 1e-12:
            return np.zeros_like(s)
        return (s - s_min) / (s_max - s_min)
    
    ucb_norm = normalize(ucb_score)
    ei_norm = normalize(ei_score)
    
    score = w_ucb * ucb_norm + w_ei * ei_norm
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
