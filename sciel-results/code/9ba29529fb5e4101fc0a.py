"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: hybrid EI + UCB with adaptive weighting.
    Balances exploitation (EI) and exploration (UCB) while accounting for evaluation cost.
    Returns 2000 finite scores (higher = evaluate next).
    """
    t_start = time.time()
    
    # Convert inputs to arrays and ensure finite values
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 1e-9), cost, 1.0)
    
    # Avoid division by zero
    sigma = np.maximum(sigma, 1e-6)
    
    # Compute z-score for expected improvement
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)  # numerical stability
    
    # Expected improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Upper Confidence Bound: mu - beta * sigma (we want to minimize, so lower is better)
    # For minimization, UCB for next eval should favor low mu and high sigma
    # ucb = mu - beta * sigma (lower ucb = more promising for minimization)
    # But we want higher score = more promising, so: score_ucb = -mu + beta * sigma
    beta = 0.1  # exploration-exploitation balance
    ucb_score = -mu + beta * sigma
    
    # Adaptive weighting: use more EI when we're close to optimum, more UCB when uncertain
    # Estimate "closeness" by how small f_best is relative to scale
    # If f_best is very small, we're likely near optimum -> favor EI
    # If f_best is large, we're still exploring -> favor UCB
    # Normalize f_best to [0,1] range for weighting
    f_best_abs = abs(f_best)
    # Assume typical scale: if f_best < 0.1, we're close; if > 1.0, we're far
    closeness = np.clip(1.0 - f_best_abs, 0, 1)
    w_ei = 0.3 + 0.7 * closeness  # weight for EI: 0.3 to 1.0
    w_ucb = 1.0 - w_ei            # weight for UCB: 0.7 to 0.0
    
    # Combine EI and UCB
    combined = w_ei * ei + w_ucb * ucb_score
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    # Use a soft cost penalty: cost^(0.5) instead of linear to be less aggressive
    cost_penalty = np.power(cost, 0.5)
    scores = combined / cost_penalty
    
    # Add a small bonus for points with very low predicted values (likely near minimum)
    # This helps when both EI and UCB are similar
    low_mu_bonus = np.where(mu < 0.01, 0.01, 0.0)
    scores = scores + low_mu_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check if we're approaching the deadline
    elapsed = time.time() - t_start
    if elapsed > 320:  # Safety margin before 330s limit
        return scores
    
    return scores
# EVOLVE-BLOCK-END
