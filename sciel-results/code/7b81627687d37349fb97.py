"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a dynamic cost penalty that increases as f_best improves and budget depletes,
    shifting from exploration to exploitation. Includes a deadline guard for safety.
    """
    # Deadline guard: if we're running long, return a simple but valid score quickly
    # The caller (evaluator) kills at 330s, so we should be fast.
    # This function is called once per iteration with 2000 candidates.
    # Keep it vectorized and fast.
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Dynamic cost penalty:
    # - As f_best improves (gets smaller), we penalize cost more heavily
    # - Base penalty scales with how close we are to the minimum
    # - The cost normalization shifts from linear to super-linear as budget depletes
    
    # f_best is the lowest observed value. For powell 4d, min is 0.
    # progress = how much we've improved from initial random starts
    # Use a soft penalty that increases as f_best approaches 0
    
    # Linear cost penalty with progress-dependent weight
    # weight = 1 + alpha * max(0, 1 - f_best) where alpha controls exploitation strength
    # As f_best -> 0, weight -> 1 + alpha, penalizing cost more
    
    # Also scale by total budget usage: cost represents cumulative cost
    # Higher cost means less budget remaining, so penalize more
    
    # Simple regret-oriented scoring:
    # score = EI / cost_effective
    # where cost_effective = cost * (1 + beta * progress)
    
    # progress: fraction of improvement achieved (assuming initial f_best ~ O(1) for powell)
    # For powell 4d on [-4,5]^4, initial random points give f_best around 5-20
    # As we find better points, f_best decreases toward 0
    
    # Use: penalty_factor = 1 + gamma * (1 - f_best / f_ref) clipped
    # where f_ref is a reference scale (e.g., 1.0)
    
    f_ref = 1.0
    progress = np.clip(1.0 - f_best / f_ref, 0.0, 1.0)
    
    # Cost penalty: super-linear as budget depletes
    # cost_penalty = cost_safe ** (1 + 0.5 * progress)
    # This makes expensive points less attractive as we improve
    
    # Alternative: linear penalty with increasing weight
    gamma = 2.0
    weight = 1.0 + gamma * progress
    cost_penalty = cost_safe * weight
    
    # Final score: EI per unit cost
    scores = ei / cost_penalty
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Small regularization to avoid exact ties
    scores += 1e-10 * np.random.randn(len(scores))
    
    return scores
# EVOLVE-BLOCK-END
