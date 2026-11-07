import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of rosenbrock 2d.
    Uses a hybrid of Expected Improvement and Probability of Improvement,
    with a cost-aware penalty that adapts based on the current best value.
    """
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Calculate z-score
    z = (f_best - mu) / sigma
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Adaptive cost-aware weighting
    # When f_best is close to 0 (good), we can be more cost-sensitive
    # When f_best is far from 0 (bad), we should explore more aggressively
    
    # Normalize f_best to get a sense of progress
    # f_best is in objective units; rosenbrock min is 0
    # Use a sigmoid-like function to blend between EI and PI
    
    # If f_best is small (close to optimum), weight more towards cost efficiency
    # If f_best is large (far from optimum), weight more towards exploration
    
    # Scale factor: as f_best approaches 0, alpha increases (more cost-aware)
    # f_best can be quite large initially, so use a robust scaling
    f_best_abs = np.abs(f_best)
    
    # Alpha: how much to weight cost-awareness (0 to 1)
    # When f_best is near 0, alpha ~ 1 (very cost-aware)
    # When f_best is large, alpha ~ 0.3 (less cost-aware, more exploration)
    alpha = 0.3 + 0.7 * np.exp(-f_best_abs / 5.0)
    
    # Cost penalty: use a power that balances exploration and exploitation
    # Lower power = less cost penalty = more exploration
    # Higher power = more cost penalty = more exploitation
    
    # Base cost exponent: starts at 0.5 (moderate penalty)
    # As we get closer to optimum, increase the exponent to be more cost-sensitive
    cost_exp = 0.5 + 0.3 * np.exp(-f_best_abs / 3.0)
    
    # Cost penalty term
    cost_penalty = np.power(cost, cost_exp)
    
    # Hybrid acquisition: blend of EI and PI, both divided by cost penalty
    # EI is in objective units, PI is a probability [0,1]
    # Scale PI to be comparable to EI
    # Use sigma as a scale for PI
    pi_scaled = pi * sigma
    
    # Weighted combination
    acq = (alpha * ei + (1 - alpha) * pi_scaled) / cost_penalty
    
    # Add a small exploration bonus for high uncertainty
    # This helps escape local optima
    exploration_bonus = 0.1 * sigma / cost_penalty
    
    acq = acq + exploration_bonus
    
    # Ensure all values are finite
    acq = np.where(np.isfinite(acq), acq, 0.0)
    
    return acq
