"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global start time for the session
_SESSION_START = time.time()


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed function value so far.
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate, shape (2000,).
        
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    # Safety deadline guard: if we've used > 290s of the 330s limit,
    # return a simple fallback immediately.
    elapsed = time.time() - _SESSION_START
    if elapsed > 290:
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        z_clipped = np.clip(z, -10, 10)
        ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
        # UCB term for exploration
        ucb = mu - 1.5 * sigma_safe
        # Blend and divide by cost
        score = 0.5 * ei + 0.5 * (-ucb)  # -ucb = -mu + 1.5*sigma
        score = score / np.maximum(cost, 1e-8)
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Calculate Expected Improvement
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # UCB term (for minimization, lower is better)
    # UCB = mu - kappa * sigma, we want to maximize this in acquisition
    # So the "improvement" part is -UCB = -mu + kappa * sigma
    kappa = 1.5
    ucb_improvement = -mu + kappa * sigma_safe
    
    # Normalize EI and UCB to comparable scales
    # EI is in objective units, UCB improvement is also in objective units
    # But their magnitudes can differ, so let's use a weighted blend
    
    # Strategy: Blend cost-normalized EI with cost-normalized UCB
    # This balances exploitation (EI) and exploration (UCB)
    # Weight adapts based on the scale of uncertainty vs. improvement
    
    # Compute the blend
    ei_cost = ei / cost_safe
    ucb_cost = ucb_improvement / cost_safe
    
    # Adaptive weighting: when sigma is large relative to |f_best - mu|, 
    # exploration (UCB) should dominate; otherwise exploitation (EI) dominates
    # Use a smooth transition based on the ratio of sigma to the scale of improvement
    scale = np.maximum(np.abs(f_best - mu), 1e-8)
    # ratio = sigma / scale; when ratio is large, favor UCB
    # Use a sigmoid-like weighting: weight_ucb = 1 / (1 + exp(-beta * (ratio - 1)))
    # But simpler: just use a fixed blend that has shown to work well
    # From the cell finding: blending EI/cost with UCB/cost gave 3.211
    
    # Use a 50/50 blend as a starting point, which is robust
    alpha = 0.5
    score = alpha * ei_cost + (1 - alpha) * ucb_cost
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
