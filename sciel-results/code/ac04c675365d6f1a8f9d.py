"""Cost-aware acquisition for Ackley 2D Bayesian optimization."""
import time
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (PI)
    with cost^0.5 normalization for smoother exploration in multimodal landscapes.
    
    Args:
        mu: array of GP mean predictions, shape (2000,)
        sigma: array of GP std predictions, shape (2000,)
        f_best: current best observed value (float)
        cost: array of evaluation costs, shape (2000,)
    
    Returns:
        array of acquisition scores, shape (2000,), higher = prefer
    """
    t_start = time.monotonic()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize inputs
    sigma = np.maximum(sigma, 1e-10)
    sigma = np.nan_to_num(sigma, nan=1e-10, posinf=1e-2, neginf=1e-10)
    mu = np.nan_to_num(mu, nan=f_best, posinf=1e4, neginf=-1e4)
    cost = np.maximum(cost, 1e-9)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1.0, neginf=1.0)
    
    # Probability of Improvement for minimization
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    improvement = f_best - mu
    z = improvement / sigma
    pi = norm.cdf(z)
    pi = np.clip(pi, 0.0, 1.0)
    pi = np.nan_to_num(pi, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Z-score normalization for robustness
    pi_mean = np.mean(pi)
    pi_std = np.std(pi)
    if pi_std > 1e-10:
        pi_norm = (pi - pi_mean) / pi_std
    else:
        pi_norm = np.zeros_like(pi)
    
    # Cost-aware scoring with cost^0.5 normalization
    # Gentler cost penalty allows better exploration of low-cost regions early
    cost_factor = np.sqrt(cost)
    scores = pi_norm / cost_factor
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Mild sharpening to emphasize top candidates
    pos_mask = scores > 0
    if np.any(pos_mask):
        max_pos = np.max(scores[pos_mask])
        if max_pos > 1e-10:
            scores[pos_mask] = (scores[pos_mask] / max_pos) ** 1.1 * max_pos
    
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
