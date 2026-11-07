"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware acquisition: combine exploration (UCB) and exploitation (EI)
    # normalized by cost to favor cheap evaluations.
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases: zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB component: encourages exploration where uncertainty is high
    # Use a moderate kappa that balances exploration/exploitation
    kappa = 1.5
    ucb = mu + kappa * sigma_safe
    
    # EI component: expected improvement over f_best
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # Clamp EI to be non-negative (EI should be >= 0)
    ei = np.maximum(ei, 0.0)
    
    # Combine UCB and EI with a blend
    # UCB gives higher values where uncertainty is high (exploration)
    # EI gives higher values where improvement is likely (exploitation)
    # Scale them to comparable ranges before combining
    # Use a weighted sum: 0.5 * normalized_uci + 0.5 * normalized_ei
    # But we need to handle the scale difference. Let's use a different approach.
    
    # Actually, let's use a simpler cost-aware approach:
    # Score = (mu + kappa * sigma) / cost^gamma for exploration
    # or Score = EI / cost^gamma for exploitation
    # Let's blend: score = (alpha * ucb + (1-alpha) * (f_best - ei)) / cost^gamma
    # Wait, EI is already the expected improvement. Let me think again.
    
    # Better approach: use a cost-normalized combination
    # The key insight: we want to find points that are both promising (low mu, high sigma)
    # and cheap (low cost).
    
    # Let's use: score = (kappa * sigma + max(0, f_best - mu)) / cost^gamma
    # This rewards high uncertainty and potential improvement, penalized by cost.
    
    gamma = 2.0
    cost_safe = np.maximum(cost, 1e-9)
    cost_penalty = cost_safe ** gamma
    
    # Exploration term: high where sigma is high
    exploration = kappa * sigma_safe
    
    # Exploitation term: high where mu is much lower than f_best
    exploitation = np.maximum(f_best - mu, 0.0)
    
    # Combine
    score = (exploration + exploitation) / cost_penalty
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
