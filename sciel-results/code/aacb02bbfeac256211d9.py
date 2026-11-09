import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    sigma = np.maximum(sigma, 1e-9)
    
    # For minimization, we want low values
    # UCB for minimization: mu - kappa * sigma (we want this to be low)
    # Score (higher is better): -(mu - kappa * sigma) = kappa * sigma - mu
    
    # Adaptive kappa: start high, decrease as we find better solutions
    # f_best tells us how close we are to the minimum (which is 0)
    # When f_best is large, we're far from min, explore more
    # When f_best is small, we're close, exploit more
    
    # Normalize f_best: if f_best is close to 0, kappa should be small
    # Use: kappa = base_kappa * (1 + f_best / (f_best + 1))
    # When f_best >> 1: kappa ≈ 2 * base_kappa
    # When f_best << 1: kappa ≈ base_kappa
    
    base_kappa = 1.5
    kappa = base_kappa * (1 + f_best / (f_best + 1.0))
    
    ucb_score = kappa * sigma - mu
    
    # EI for minimization
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware: divide by cost
    # But also add a small exploration bonus for high sigma
    
    # Final score: blend of UCB/cost and EI/cost
    ucb_cost = ucb_score / cost
    ei_cost = ei / cost
    
    # Weight based on how much we've learned
    # Use the standard deviation of mu as a proxy for learning
    # Early: high uncertainty, use more UCB
    # Late: low uncertainty, use more EI
    
    # Simple blend: 50-50, but adapt based on f_best
    # When f_best is high (early), weight UCB more
    weight_ucb = 0.3 + 0.4 * (f_best / (f_best + 0.5))
    weight_ucb = np.clip(weight_ucb, 0, 1)
    
    return weight_ucb * ucb_cost + (1 - weight_ucb) * ei_cost
