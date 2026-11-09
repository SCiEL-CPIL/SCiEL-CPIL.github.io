"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Lower Confidence Bound (LCB) acquisition function with cost awareness.
    
    For minimization, LCB is:
        LCB(x) = mu(x) - beta * sigma(x)
    
    This is a fundamentally different design from EI/KG:
    - EI/KG are "improvement-based" (reward for beating f_best)
    - LCB is "confidence-based" (reward for low predicted value + high uncertainty)
    
    For the Griewank function with its flat landscape, LCB naturally:
    1. Explores regions with high uncertainty (high sigma)
    2. Exploits regions with low predicted values (low mu)
    3. The beta parameter controls the exploration/exploitation tradeoff
    
    Cost-awareness: divide by cost to prefer cheaper evaluations.
    We use a soft cost penalty that's less aggressive than linear division.
    """
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # LCB for minimization: lower is better
    # beta controls exploration: higher beta = more exploration
    # For Griewank's flat landscape, we want moderate exploration
    beta = 1.5
    
    lcb = mu - beta * sigma_safe
    
    # Cost-aware adjustment: prefer lower cost
    # Use a softer penalty than 1/cost to avoid over-penalizing
    # cost is in [1, 2], so 1/cost is in [0.5, 1]
    # Use a power to make the penalty less extreme
    cost_penalty = cost ** 0.5  # sqrt(cost) is in [1, sqrt(2)] ~ [1, 1.414]
    
    # Score: higher is better for evaluation priority
    # We want to maximize (f_best - LCB) / cost_penalty
    # But LCB can be negative, so we need to be careful
    # Instead, use: score = (f_best - mu) / sigma_safe * (1 + beta) - beta * sigma_safe / cost_penalty
    # This is equivalent to LCB-based scoring with cost
    
    # Simpler approach: normalize LCB and divide by cost
    # Since we're minimizing, lower LCB is better
    # Score = -LCB / cost_penalty (higher score = lower LCB = better)
    
    # But LCB can be very negative, so let's use a bounded version
    # The expected improvement interpretation:
    # We want to find points where f(x) < f_best
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # E[max(0, f_best - f(x))] = EI
    
    # For LCB-based approach, use:
    # score = P(f(x) < f_best) * (f_best - mu) / cost + exploration_bonus
    z = (f_best - mu) / sigma_safe
    prob_improve = norm.cdf(z)
    expected_improvement = (f_best - mu) * prob_improve
    
    # Exploration bonus: reward high uncertainty regions
    # Use the probability that the point is significantly better than f_best
    # P(f(x) < f_best - delta) for some delta
    delta = 0.01  # characteristic scale for Griewank
    z_delta = (f_best - delta - mu) / sigma_safe
    prob_significant = norm.cdf(z_delta)
    
    # Combined score: exploitation + exploration
    # Exploitation: expected improvement
    # Exploration: probability of finding a significantly better point
    score = expected_improvement / cost_penalty + 0.5 * prob_significant * sigma_safe / cost_penalty
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
