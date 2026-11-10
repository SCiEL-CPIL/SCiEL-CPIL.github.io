"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multi-constraint knapsack problem.
    
    Uses a combination of:
    1. Total prize (higher is better)
    2. Feasibility margin: how easily the item fits into the tightest constraint
    3. Density: prize per unit weight across constraints
    
    The heuristic is designed to prioritize items that:
    - Have high total prize
    - Fit easily into all constraints (high slack relative to capacity)
    - Have good prize-to-weight ratio
    """
    t0 = time.time()
    
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Ensure correct shapes
    if prizes.shape != (5, 100) or weights.shape != (5, 100):
        # Fallback: return a simple valid array
        return np.ones(100, dtype=np.float64)
    
    # Ensure positive values to avoid division issues
    prizes = np.maximum(prizes, 1e-9)
    weights = np.maximum(weights, 1e-9)
    
    capacity = 25.0
    n_items = 100
    n_constraints = 5
    
    # Total prize per item (sum across constraints)
    total_prize = prizes.sum(axis=0)  # shape (100,)
    
    # For each item, compute the minimum remaining capacity ratio across constraints
    # If item i is added, constraint c has remaining capacity (capacity - weights[c, i])
    # We want to prioritize items where this minimum ratio is highest (fits most easily)
    # slack_ratio[c, i] = (capacity - weights[c, i]) / weights[c, i]
    # But this can be negative if weights > capacity, so we handle that
    slack_ratio = (capacity - weights) / weights  # shape (5, 100)
    
    # For items where weight > capacity, slack_ratio is negative; clamp to a small negative
    slack_ratio = np.maximum(slack_ratio, -1.0)
    
    # Minimum slack ratio across constraints for each item (bottleneck constraint)
    min_slack = slack_ratio.min(axis=0)  # shape (100,)
    
    # We want to reward items with high min_slack (fit easily) and penalize those with low/negative
    # Use a smooth transformation: exp(min_slack) gives higher values for items that fit easily
    # But if min_slack is very negative, exp is near 0, which is fine (low attractiveness)
    # If min_slack is very positive, exp is large, which rewards easy-to-fit items
    feasibility_score = np.exp(np.clip(min_slack, -5, 5))  # shape (100,)
    
    # Density: total prize / total weight (across all constraints)
    total_weight = weights.sum(axis=0)  # shape (100,)
    density = total_prize / (total_weight + 1e-9)  # shape (100,)
    
    # Normalize density to [0, 1] range for better scaling
    density_min = density.min()
    density_max = density.max()
    if density_max > density_min:
        density_norm = (density - density_min) / (density_max - density_min)
    else:
        density_norm = np.ones_like(density)
    
    # Normalize total prize to [0, 1]
    prize_min = total_prize.min()
    prize_max = total_prize.max()
    if prize_max > prize_min:
        prize_norm = (total_prize - prize_min) / (prize_max - prize_min)
    else:
        prize_norm = np.ones_like(total_prize)
    
    # Combine: prize * feasibility * (0.5 + 0.5 * density)
    # This rewards high prize, good feasibility, and good density
    heuristic = prize_norm * feasibility_score * (0.5 + 0.5 * density_norm)
    
    # Ensure all values are positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    # Check deadline
    if time.time() - t0 > 290.0:
        # Safety: return what we have
        return heuristic
    
    return heuristic
# EVOLVE-BLOCK-END
