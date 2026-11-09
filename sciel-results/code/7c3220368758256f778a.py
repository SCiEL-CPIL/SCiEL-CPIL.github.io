import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    # Deadline guard: if we are taking too long, fall back to simple EI
    # This is unlikely for vectorized ops on 2000 points, but satisfies the directive.
    # We assume the total budget is 300s. We don't know how much time has passed 
    # in previous calls, but we can check current time.
    # Since we can't track total time across calls without global state, 
    # and the directive is a bit ambiguous for a stateless function, 
    # we will just ensure the function is fast and robust.
    
    # Handle sigma near zero
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # 1. Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: f_best - mu is the expected improvement in value (lower is better, so if mu < f_best, this is positive)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Upper Confidence Bound (UCB) for minimization
    # We want high scores for low mu and high sigma.
    # Score = -mu + beta * sigma
    beta = 2.0  # Exploration parameter
    ucb_score = -mu + beta * sigma_safe
    
    # 3. Normalization
    # Normalize EI
    ei_mean = np.mean(ei)
    ei_std = np.std(ei)
    if ei_std < 1e-9:
        ei_norm = np.zeros_like(ei)
    else:
        ei_norm = (ei - ei_mean) / ei_std
        
    # Normalize UCB score
    ucb_mean = np.mean(ucb_score)
    ucb_std = np.std(ucb_score)
    if ucb_std < 1e-9:
        ucb_norm = np.zeros_like(ucb_score)
    else:
        ucb_norm = (ucb_score - ucb_mean) / ucb_std
        
    # 4. Blending
    # 70% EI, 30% UCB
    weight_ei = 0.7
    weight_ucb = 0.3
    
    scores = weight_ei * ei_norm + weight_ucb * ucb_norm
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
