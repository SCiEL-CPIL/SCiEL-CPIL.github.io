import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    Uses a hybrid EI + UCB strategy with adaptive weighting.
    
    Parameters:
    mu (np.array): Posterior mean, shape (2000,)
    sigma (np.array): Posterior standard deviation, shape (2000,)
    f_best (float): Best observed function value so far.
    
    Returns:
    np.array: Scores for the candidates, shape (2000,). Higher is better.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate Z-score for Expected Improvement
    z = (f_best - mu) / sigma
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB score for exploration: mu - beta * sigma (for minimization, we want low mu and high sigma)
    # Actually for minimization with UCB: score = mu - beta * sigma (lower mu is better, higher sigma encourages exploration)
    # But we want higher score = better, so: score = -mu + beta * sigma
    # Or equivalently: UCB_score = -mu + beta * sigma
    
    # Adaptive beta: start high for exploration, decrease as f_best approaches 0
    # Weierstrass minimum is 0, so as f_best -> 0, we should exploit more
    if f_best > 0.1:
        beta = 2.5  # aggressive exploration
    elif f_best > 0.01:
        beta = 1.5  # moderate exploration
    else:
        beta = 0.5  # exploitation
    
    ucb_score = -mu + beta * sigma
    
    # Normalize both to [0, 1] range for combination
    ei_min, ei_max = np.min(ei), np.max(ei)
    ucb_min, ucb_max = np.min(ucb_score), np.max(ucb_score)
    
    if ei_max - ei_min > 1e-12:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.zeros_like(ei)
    
    if ucb_max - ucb_min > 1e-12:
        ucb_norm = (ucb_score - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb_score)
    
    # Adaptive weight: more EI as f_best gets closer to 0
    if f_best > 0.1:
        w_ei = 0.3
    elif f_best > 0.01:
        w_ei = 0.5
    else:
        w_ei = 0.7
    
    w_ucb = 1.0 - w_ei
    
    score = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
