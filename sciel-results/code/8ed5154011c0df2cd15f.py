"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def _soft_rank(x, temperature=1.0):
    """
    Soft ranking using log-sum-exp for smooth gradients.
    Returns values in [0, 1] where higher = better rank.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n <= 1:
        return np.ones_like(x)
    
    # Normalize to avoid overflow in exp
    x_max = np.max(x)
    x_shifted = x - x_max
    
    # Compute soft ranks via log-sum-exp
    # soft_rank_i = log(1 + sum_j exp((x_j - x_i)/T))
    # This gives a smooth approximation to discrete ranks
    
    # For efficiency, use a simpler approach: sigmoid-based soft ranking
    # score_i = 1 / (1 + exp(-(x_i - median)/T))
    median = np.median(x)
    scores = 1.0 / (1.0 + np.exp(-(x - median) / temperature))
    
    # Normalize to [0, 1]
    s_min = np.min(scores)
    s_max = np.max(scores)
    if s_max - s_min > 1e-12:
        scores = (scores - s_min) / (s_max - s_min)
    else:
        scores = np.full_like(scores, 0.5)
    
    return scores

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a smooth quantile-based probability-of-improvement heuristic:
    - Ranks candidates by posterior probability of improvement normalized by cost
    - Uses soft-min/soft-max surrogates for smooth cost normalization
    - Adds uncertainty-aware exploration bonus
    - Decouples search from noisy mean estimates with smooth gradients
    
    Args:
        mu: (2000,) array of GP mean predictions.
        sigma: (2000,) array of GP standard deviations.
        f_best: scalar, the best observed value so far.
        cost: (2000,) array of costs for each candidate.
        
    Returns:
        (2000,) array of scores. Higher is better.
    """
    start_time = time.time()
    
    # Ensure arrays are float
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Prevent division by zero in sigma
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Prevent division by zero in cost
    cost_safe = np.maximum(cost, 1e-10)
    
    # Calculate Z-score for P(f(x) < f_best)
    # Since we are minimizing, improvement is when f(x) is lower than f_best.
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Clip PI to [0, 1] for numerical stability
    pi = np.clip(pi, 0.0, 1.0)
    
    # Compute cost-normalized PI using soft-min surrogate
    # Instead of hard pi/cost, use a smooth approximation
    # soft_cost_inv = 1 / (cost + epsilon) with smoothing
    cost_smooth = cost_safe + 0.1  # Add regularization to smooth the cost normalization
    pi_cost = pi / cost_smooth
    
    # Add uncertainty-aware exploration bonus
    # Higher sigma means more uncertainty, which can be valuable for exploration
    # But we weight it by how close we are to the boundary (z near 0)
    sigma_norm = sigma_safe / (np.max(sigma_safe) + 1e-10)
    
    # Exploration bonus: encourage points with high uncertainty AND moderate improvement probability
    # This helps navigate the Rosenbrock valley by exploring uncertain regions
    exploration_bonus = sigma_norm * (1.0 - np.abs(z) / (np.max(np.abs(z)) + 1e-10))
    exploration_bonus = np.clip(exploration_bonus, 0.0, 1.0)
    
    # Combine exploitation (pi_cost) and exploration (bonus)
    # Weight: 70% exploitation, 30% exploration
    combined = 0.7 * pi_cost + 0.3 * exploration_bonus * np.max(pi_cost)
    
    # Use soft ranking for smooth score assignment
    # Temperature controls how sharp the ranking is
    temperature = 0.5
    score_soft = _soft_rank(combined, temperature=temperature)
    
    # Also compute discrete rank for comparison
    ranks = np.argsort(np.argsort(combined))
    n = len(combined)
    score_ranks = ranks.astype(float) / max(n - 1, 1)
    
    # Normalize combined to [0, 1]
    c_min = np.min(combined)
    c_max = np.max(combined)
    if c_max - c_min > 1e-12:
        combined_norm = (combined - c_min) / (c_max - c_min)
    else:
        combined_norm = np.zeros_like(combined)
    
    # Blend: 50% soft rank, 30% discrete rank, 20% magnitude
    scores = 0.5 * score_soft + 0.3 * score_ranks + 0.2 * combined_norm
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 280:
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
