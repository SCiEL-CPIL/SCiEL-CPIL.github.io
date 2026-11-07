"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness matrix for ant colony optimization.
    
    The heuristic uses a 'regret' or opportunity-cost based metric:
    For edge (i, j), the attractiveness is based on:
    1. The prize of node j (higher is better)
    2. The total cost of visiting j and returning to start: d[i,j] + d[j,0]
    3. A penalty based on the maximum prize of other reachable nodes from i
       within the remaining budget, to bias away from low-opportunity-cost choices
    
    Args:
        prizes: array of shape (n,) with prize values, prizes[0] = 0
        distance_matrix: array of shape (n, n) with pairwise distances
        max_len: maximum allowed path length (3 in this task)
    
    Returns:
        heuristic: array of shape (n, n) with edge attractiveness
    """
    n = len(prizes)
    d = np.asarray(distance_matrix, dtype=np.float64)
    prizes = np.asarray(prizes, dtype=np.float64)
    
    # Cost of going from i to j and back to 0
    # total_cost[i, j] = d[i, j] + d[j, 0]
    return_cost = d[:, 0]  # shape (n,)
    total_cost = d + return_cost[None, :]  # shape (n, n)
    
    # For each node i, determine which nodes j are reachable within max_len
    # An ant at i can go to j if current_length + d[i,j] + d[j,0] <= max_len
    # Since we don't know current_length at heuristic computation time,
    # we use a conservative estimate: assume minimal remaining budget
    
    # Base attractiveness: prize / cost
    eps = 1e-9
    base_attr = (prizes[None, :] + eps) / (total_cost + eps)  # shape (n, n)
    
    # Regret metric: for each i, find the max prize among nodes reachable from i
    # that are NOT j, to penalize choices that forgo high-prize alternatives
    
    # For each source node i, compute the maximum prize of other nodes
    # that could also be reached (within a reasonable budget)
    max_prize = prizes.max()
    
    # For each i, the regret of choosing j is how much prize we forgo
    # compared to the best alternative reachable from i
    
    # Compute for each i, the max prize among all j' != j that are reachable
    # We'll use a vectorized approach:
    # For each i, create a mask of reachable nodes (total_cost[i, :] <= max_len)
    # Then for each j, the regret is max_prize_reachable - prizes[j]
    
    heuristic = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        # Nodes reachable from i within max_len (conservative: total_cost <= max_len)
        reachable_mask = total_cost[i, :] <= max_len
        if not np.any(reachable_mask):
            heuristic[i, :] = eps
            continue
        
        reachable_prizes = prizes[reachable_mask].copy()
        max_reachable_prize = reachable_prizes.max()
        
        # For each j, compute regret-based attractiveness
        for j in range(n):
            if not reachable_mask[j]:
                heuristic[i, j] = eps
                continue
            
            # Opportunity cost: how much better could we do?
            # If j is the best reachable, regret is 0 (or low)
            # If j is much worse than the best, regret is high
            regret = max_reachable_prize - prizes[j]
            
            # Attractiveness: high prize, low cost, low regret
            # Use: prize * (1 + regret_penalty) / cost
            # where regret_penalty reduces attractiveness when regret is high
            # Actually, we want to AVOID high regret, so:
            # attractiveness = prize / cost * (1 / (1 + regret))
            # This way, if regret is 0 (j is the best), factor is 1
            # If regret is large, factor is small
            
            regret_factor = 1.0 / (1.0 + regret)
            heuristic[i, j] = base_attr[i, j] * regret_factor
    
    # Ensure all values are positive and finite
    heuristic[~np.isfinite(heuristic)] = eps
    heuristic[heuristic <= 0] = eps
    
    return heuristic
# EVOLVE-BLOCK-END
