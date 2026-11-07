"""Rewritten acquisition with hybrid EI/UCB cost-aware scoring and time guard."""
import time
import numpy as np
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin before 300s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Hybrid EI/UCB per unit cost.
    
    Args:
        mu: array shape (2000,) - GP mean predictions in objective units
        sigma: array shape (2000,) - GP std dev predictions in objective units
        f_best: float - lowest observed value
        cost: array shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Time guard: if we're running out of time, return a simple fallback
    if time.time() > _DEADLINE:
        return np.ones_like(mu) / np.maximum(cost, 1e-9)
    
    eps = 1e-9
    s = np.maximum(sigma, eps)
    c = np.maximum(cost, eps)
    
    # 1. Expected Improvement
    z = (f_best - mu) / s
    z = np.clip(z, -10.0, 10.0)
    ei = (f_best - mu) * norm.cdf(z) + s * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # 2. UCB (Lower Confidence Bound for minimization)
    # We want to minimize, so we look for low mu and high sigma (uncertainty).
    # LCB = mu - beta * sigma. We maximize -LCB = -mu + beta * sigma.
    # Choose beta based on typical exploration needs.
    beta = 1.0 
    ucb = -mu + beta * s
    
    # Combine EI and UCB
    # EI is better for exploitation, UCB for exploration.
    # Weight them.
    hybrid = 0.7 * ei + 0.3 * ucb
    
    # Cost-aware scaling
    # Divide by cost to favor cheaper points.
    # Using a power of cost can adjust sensitivity.
    # cost^1 is standard. cost^0.5 is less sensitive.
    # Let's try standard division first, but ensure stability.
    score = hybrid / c
    
    # Add a small exploration boost for high uncertainty relative to cost
    # This helps escape local minima if EI is too greedy.
    score += 0.1 * (s / c)
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=1e9, neginf=-1e9)
    
    return score
