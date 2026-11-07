"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items using a scarcity-weighted marginal utility heuristic.
    
    The heuristic rewards items that offer high prize relative to the scarcity of
    the specific resources they consume. Scarcity is estimated by comparing the
    top-k efficiency of items per constraint against the mean efficiency.
    
    A soft-max bottleneck penalty (log-sum-exp) smooths the min-objective to avoid
    discontinuities in the heuristic landscape.
    """
    start_time = time.time()
    DEADLINE = 325.0  # Safety margin before 330s kill
    
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Validate shapes
    if prizes.shape != (5, 100) or weights.shape != (5, 100):
        raise ValueError("Invalid shape")
    
    if not (np.all(np.isfinite(prizes)) and np.all(np.isfinite(weights))):
        raise ValueError("Non-finite values")
    
    n_constraints, n_items = prizes.shape
    
    # Compute per-item, per-constraint efficiency: prize / weight for each constraint
    # Avoid division by zero
    weights_safe = np.maximum(weights, 1e-12)
    efficiency = prizes / weights_safe  # shape (n_constraints, n_items)
    
    # For each constraint, compute scarcity measure:
    # Sort efficiencies in descending order, take top-k
    # Scarcity ratio = top-k mean / overall mean (higher means more concentrated)
    k = max(1, n_items // 10)  # top 10% of items
    
    scarcity_weights = np.ones(n_constraints)
    
    for c in range(n_constraints):
        eff_c = efficiency[c]
        mean_eff = np.mean(eff_c)
        if mean_eff < 1e-12:
            continue
        # Top-k efficiencies
        top_k_eff = np.sort(eff_c)[::-1][:k]
        top_k_mean = np.mean(top_k_eff)
        # Scarcity ratio: how much more efficient the top items are vs average
        scarcity_ratio = top_k_mean / (mean_eff + 1e-12)
        scarcity_weights[c] = scarcity_ratio
    
    # Normalize scarcity weights
    scarcity_weights = scarcity_weights / (np.sum(scarcity_weights) + 1e-12)
    
    # Compute per-item weighted scarcity cost:
    # For each item, the "cost" is the weighted sum of its resource consumption
    # weighted by scarcity
    # An item that uses scarce resources heavily should be penalized
    
    # Per-item resource usage vector (normalized by capacity)
    capacity = 25.0
    usage = weights / capacity  # shape (n_constraints, n_items), values in [0, ~1]
    
    # Weighted scarcity cost per item: sum over constraints of scarcity_weight * usage
    scarcity_cost = np.sum(scarcity_weights[:, None] * usage, axis=0)  # shape (n_items,)
    
    # Base attractiveness: prize / (1 + scarcity_cost)
    # Items with high prize and low scarcity cost are more attractive
    base_attractiveness = prizes[0] / (1.0 + scarcity_cost)
    
    # Alternative: use mean prize across constraints as a robustness measure
    mean_prize = np.mean(prizes, axis=0)
    base_attractiveness = mean_prize / (1.0 + scarcity_cost)
    
    # Soft-max bottleneck penalty:
    # For each item, compute the minimum "slack" across constraints
    # slack_c = capacity - weight_c (how much room left)
    # We want to penalize items that would leave little slack in tight constraints
    
    # Estimate tightness: for each constraint, what fraction of items can fit?
    # If few items can fit in a constraint, it's tight
    
    # Compute per-constraint tightness
    tightness = np.ones(n_constraints)
    for c in range(n_constraints):
        # Number of items that individually fit in this constraint
        num_fit = np.sum(weights[c] <= capacity)
        tightness[c] = n_items / (num_fit + 1e-12)
    
    # Normalize tightness
    tightness = tightness / (np.sum(tightness) + 1e-12)
    
    # Log-sum-exp soft-max of negative slacks (penalizes items that consume tight resources)
    # For each item, compute -log-sum-exp(-tightness * usage)
    # This smoothly penalizes items that use resources in tight constraints
    
    # Compute soft penalty
    # penalty_c = tightness[c] * usage[c] for each constraint
    # soft_penalty = log-sum-exp(penalty) across constraints
    # Use log-sum-exp trick for numerical stability
    
    penalty_matrix = tightness[:, None] * usage  # shape (n_constraints, n_items)
    
    # Log-sum-exp across constraints (axis=0)
    max_penalty = np.max(penalty_matrix, axis=0)
    soft_penalty = max_penalty + np.log(np.sum(np.exp(penalty_matrix - max_penalty[:, None]), axis=0))
    
    # Final heuristic: base attractiveness reduced by soft penalty
    heuristic = base_attractiveness * np.exp(-soft_penalty)
    
    # Ensure all values are positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Return a simple fallback
        return np.ones(n_items)
    
    # Normalize to have reasonable magnitude
    heuristic = heuristic / (np.mean(heuristic) + 1e-12)
    
    return heuristic
# EVOLVE-BLOCK-END
