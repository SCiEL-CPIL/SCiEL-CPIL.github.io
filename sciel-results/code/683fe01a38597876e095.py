"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI) 
    and Upper Confidence Bound (UCB) for minimization.
    
    Args:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP standard deviations
        f_best: scalar, best observed value
        cost: (2000,) array of evaluation costs
    
    Returns:
        (2000,) array of acquisition scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # 1. Expected Improvement (for minimization)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # 2. UCB Potential (for minimization)
    # We want to maximize the chance of finding a point better than f_best.
    # The lower bound of the prediction is mu - beta * sigma.
    # The "potential improvement" is f_best - (mu - beta * sigma).
    beta = 1.5  # Exploration-exploitation trade-off
    ucb_lower_bound = mu - beta * sigma
    ucb_potential = f_best - ucb_lower_bound
    ucb_potential = np.maximum(ucb_potential, 0.0)
    
    # 3. Normalization
    # Normalize by the scale of the current best value to make scores comparable
    # across different stages of optimization (when f_best is large vs small).
    scale = abs(f_best) + 1.0
    
    ei_norm = ei / scale
    ucb_norm = ucb_potential / scale
    
    # 4. Cost-Awareness
    # Divide by cost to prefer cheaper evaluations.
    # Cost is in [1, 2], so this is a mild penalty.
    ei_cost = ei_norm / cost
    ucb_cost = ucb_norm / cost
    
    # 5. Combination
    # Weighted sum. EI is more reliable (expected value), UCB is more exploratory.
    # We favor EI slightly more to converge faster on the narrow Rosenbrock valley.
    w_ei = 0.7
    w_ucb = 0.3
    
    score = w_ei * ei_cost + w_ucb * ucb_cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
