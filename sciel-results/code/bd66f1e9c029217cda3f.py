"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard
_DEADLINE = time.time() + 320.0  # 10s safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    mu, sigma: shape (2000,) - GP mean and std (in objective units)
    f_best: scalar - lowest observed value
    cost: shape (2000,) - evaluation cost for each candidate
    
    Returns: shape (2000,) - scores (higher = evaluate next)
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against zero/negative costs
    cost = np.maximum(cost, 1e-9)
    
    # Guard against zero/negative sigma
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI)
    delta = f_best - mu
    z = delta / sigma
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    ei = delta * Phi_z + sigma * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Compute Upper Confidence Bound (UCB)
    beta = 1.0
    ucb = mu + beta * sigma
    
    # Cost-aware scoring: combine EI and UCB per unit cost
    # Use a weighted combination that balances exploitation and exploration
    ei_score = ei / cost
    ucb_score = ucb / cost
    
    # Blend: primarily EI with some UCB for exploration
    # In early iterations (high sigma), UCB helps explore
    # In later iterations (low sigma), EI dominates
    sigma_mean = np.mean(sigma)
    sigma_scale = sigma_mean / (np.max(sigma) + 1e-9)
    
    # Weight that decreases as we get more confident
    # When sigma is large relative to max, we're early => more exploration
    # When sigma is small, we're late => more exploitation
    w_ucb = 0.3 * (1.0 + sigma_scale)
    w_ei = 1.0
    
    score = (w_ei * ei_score + w_ucb * ucb_score)
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
