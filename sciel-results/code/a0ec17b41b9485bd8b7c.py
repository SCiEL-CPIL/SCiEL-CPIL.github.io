import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB exploration term
    # beta: exploration weight. Higher when uncertainty is high relative to scale.
    # Use a fixed beta that balances exploitation and exploration.
    beta = 1.0
    ucb = mu + beta * sigma_safe
    
    # Blend EI and UCB
    # When f_best is large (early), we can afford more exploration
    # When f_best is small (late), we want more exploitation
    f_best_scale = max(abs(f_best), 1e-6)
    # gamma: weight on EI vs UCB. Higher gamma = more exploitation
    gamma = 1.0 / (1.0 + f_best_scale)
    
    # Combine: score = gamma * EI + (1-gamma) * (-UCB)
    # But we want higher scores to be better, so negate UCB
    blended = gamma * ei + (1.0 - gamma) * (-ucb)
    
    # Cost weighting
    alpha = 1.0 / (1.0 + f_best_scale)
    cost_weight = cost_safe ** (-alpha)
    
    score = blended * cost_weight
    
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    return score
