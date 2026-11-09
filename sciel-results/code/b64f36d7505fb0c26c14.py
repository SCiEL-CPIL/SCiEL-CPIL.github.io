"""Rewritten program for cost-aware acquisition; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed function value so far
    cost : array of shape (2000,) - cost of evaluating each point
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = better)
    """
    # Handle edge cases where sigma might be very small or zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute z-score for Expected Improvement
    # Since we're minimizing, EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (for minimization)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring
    # Cost ranges from 1 to 2 (since mean(x scaled to [0,1]) is in [0,1])
    # We want to balance improvement against cost
    # 
    # Strategy: Use a soft cost penalty that's adaptive
    # At low costs (near 1), we allow more exploration
    # At high costs (near 2), we require more certain improvement
    #
    # score = ei / (cost^alpha) where alpha controls cost sensitivity
    # 
    # Alternatively, use a more nuanced approach:
    # - Normalize cost to [0, 1]: cost_norm = (cost - 1) / (2 - 1) = cost - 1
    # - Use a penalty that's stronger for expensive points
    #
    # Let's use: score = ei * (1 - 0.3 * (cost - 1))
    # This gives:
    #   cost=1: score = ei * 1.0
    #   cost=2: score = ei * 0.7
    #
    # Or even better, use exponential decay:
    # score = ei * exp(-beta * (cost - 1))
    # With beta = 0.5:
    #   cost=1: score = ei
    #   cost=2: score = ei * exp(-0.5) ≈ ei * 0.607
    
    # I'll use a combination: power law with moderate exponent
    # This is different from the current approach which used linear denominator
    
    # Option 1: Power law
    # scores = ei / np.power(cost, 0.8)
    
    # Option 2: Exponential
    # scores = ei * np.exp(-0.5 * (cost - 1.0))
    
    # Option 3: Hybrid - use both EI magnitude and cost
    # For very high EI, cost matters less; for low EI, cost matters more
    # This is like a risk-adjusted return
    
    # Let me try a different approach: use the ratio of EI to cost,
    # but with a floor to prevent division issues, and add a small
    # exploration bonus for uncertain regions
    
    # Base score: EI per unit cost
    base_score = ei / (cost + 1e-9)
    
    # Add uncertainty bonus: points with high sigma are worth exploring
    # even if current EI is low, because they might lead to better f_best
    # Scale the bonus by how much improvement is possible
    # The maximum possible improvement is f_best - 0 (since min is 0)
    # So uncertainty bonus = sigma * (1 - f_best/max_possible)
    # But we don't know max_possible exactly. Use a heuristic.
    
    # Actually, let's keep it simpler but more effective:
    # Use a modified EI that accounts for cost more smoothly
    
    # Final approach: 
    # 1. Compute EI
    # 2. Apply cost penalty using a smooth function
    # 3. Add a small exploration term for high-uncertainty regions
    
    # Cost penalty: use (cost)^(-gamma) with gamma ~ 0.5-1.0
    # This makes expensive points less attractive but not prohibitively so
    
    gamma = 0.7  # Cost sensitivity parameter
    cost_penalty = np.power(cost, gamma)
    
    # Primary score: EI adjusted for cost
    scores = ei / cost_penalty
    
    # Exploration bonus: encourage evaluating uncertain points
    # This helps escape local optima and explore the space
    # Weight the bonus by the relative uncertainty
    # Normalize sigma to [0, 1] approximately
    sigma_max = np.max(sigma) if np.max(sigma) > 0 else 1.0
    exploration_weight = 0.1  # How much to weight exploration
    exploration_bonus = exploration_weight * (sigma / sigma_max) * (f_best - np.min(mu))
    
    # But only add exploration bonus when EI is small
    # This prevents the exploration bonus from dominating
    ei_threshold = np.percentile(ei, 25)  # 25th percentile as threshold
    exploration_mask = ei < ei_threshold
    scores[exploration_mask] += exploration_bonus[exploration_mask]
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure non-negative scores (for safety)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
