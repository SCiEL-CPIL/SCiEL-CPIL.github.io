"""Rewritten cost-aware acquisition for Ackley 2D Bayesian optimization."""
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a blend of expected improvement
    and a UCB-style exploration term, both normalized by cost.
    
    Args:
        mu: array of GP mean predictions, shape (2000,)
        sigma: array of GP std predictions, shape (2000,)
        f_best: current best observed value
        cost: array of evaluation costs, shape (2000,)
    
    Returns:
        array of acquisition scores, shape (2000,), higher = prefer
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-12)
    sigma = np.nan_to_num(sigma, nan=1e-12, posinf=1.0, neginf=1e-12)
    mu = np.nan_to_num(mu, nan=f_best, posinf=1e6, neginf=-1e6)
    cost = np.maximum(cost, 1e-9)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1.0, neginf=1.0)
    
    # Expected Improvement for minimization
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB-style term: -mu + beta * sigma (we maximize, so lower mu and higher sigma are good)
    # Adapt beta based on the scale of sigma in the candidate set
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma)
    
    # If uncertainty is high relative to improvement potential, explore more
    # Beta between 0.5 and 2.5
    if sigma_mean > 0.01:
        beta = 1.0 + 1.5 * np.exp(-sigma_mean / 0.1)
    else:
        beta = 0.5
    
    # Clip beta to reasonable range
    beta = np.clip(beta, 0.3, 3.0)
    
    ucb_term = -mu + beta * sigma
    
    # Blend EI and UCB. EI is in objective units, UCB is also in objective units.
    # Scale them to be comparable. EI can be zero or negative (if mu > f_best).
    # We want to prefer points where EI is high AND the UCB is high.
    # Use a weighted sum, but normalize each to [0,1] roughly over the candidate set.
    
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min + 1e-9
    ei_norm = (ei - ei_min) / ei_range
    
    ucb_min = np.min(ucb_term)
    ucb_max = np.max(ucb_term)
    ucb_range = ucb_max - ucb_min + 1e-9
    ucb_norm = (ucb_term - ucb_min) / ucb_range
    
    # Weight: when f_best is close to 0 (good), rely more on EI (exploitation).
    # When f_best is far from 0, rely more on UCB (exploration).
    # Ackley min is 0, so f_best should be >= 0.
    # If f_best < 0.1, weight EI 0.7, UCB 0.3
    # If f_best > 1.0, weight EI 0.3, UCB 0.7
    if f_best < 0.05:
        w_ei = 0.8
        w_ucb = 0.2
    elif f_best < 0.2:
        w_ei = 0.6
        w_ucb = 0.4
    elif f_best < 1.0:
        w_ei = 0.4
        w_ucb = 0.6
    else:
        w_ei = 0.2
        w_ucb = 0.8
    
    combined = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Cost-aware: divide by cost
    scores = combined / cost
    
    # Ensure finite, non-negative (for safety, though scores can be negative)
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
