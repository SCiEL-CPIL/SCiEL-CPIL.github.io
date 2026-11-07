"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    """
    Build a heuristic matrix for the ant colony optimization of orienteering.
    
    The heuristic H[i, j] represents the attractiveness of moving from node i to node j.
    We use a formula that balances:
    1. The prize at node j (higher is better).
    2. The total cost of visiting j and returning to start (d[i, j] + d[j, 0]), 
       as this consumes the remaining budget.
    3. The feasibility of the move, estimated by the slack remaining after 
       assuming the shortest path to i and the direct return from j.
    """
    start_time = time.time()
    DEADLINE = start_time + 290.0  # 300s limit, keep 10s margin
    
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    n = len(prizes)
    
    # Ensure shape correctness
    if distance_matrix.shape != (n, n):
        default = np.ones((n, n), dtype=np.float64) * 1e-9
        default[0, 0] = 0.0
        return default
    
    eps = 1e-9
    
    # Precompute distances from start to all nodes (lower bound on dist_so_far)
    min_dist_from_0 = distance_matrix[0, :]
    
    # Calculate the "return cost" for each node: d[j, 0]
    return_cost = distance_matrix[:, 0]
    
    # Total cost of edge (i, j) in terms of budget consumption:
    # We pay d[i, j] to get to j, and we must eventually pay d[j, 0] to return.
    # So the "effective distance" is d[i, j] + d[j, 0].
    effective_dist = distance_matrix + return_cost[None, :]
    
    # Minimum possible tour length if we take edge (i, j) as the first and last step?
    # Actually, min_dist_from_0[i] is the min distance to reach i.
    # If we are at i, we have used at least min_dist_from_0[i].
    # Then we go to j (cost d[i, j]) and return (cost d[j, 0]).
    # Total min tour length estimate: min_dist_from_0[i] + d[i, j] + d[j, 0]
    min_tour_len = min_dist_from_0[:, None] + effective_dist
    
    # Feasibility mask: 1 if min_tour_len <= max_len, else 0
    feasible = min_tour_len <= max_len
    
    # Slack: how much budget is left
    slack = max_len - min_tour_len
    slack = np.maximum(slack, 0.0)
    
    # Base heuristic: Prize / (Effective Distance)^beta
    # We use beta=1.0 as a standard start, but might tune it.
    # Using effective_dist in denominator directly penalizes nodes far from start.
    # We add eps to avoid division by zero.
    safe_eff_dist = effective_dist + eps
    
    # Prize component: prizes[None, :]
    # We might use a power of prize to emphasize high prizes more.
    # Let's try alpha=1.0 for now, but the inspiration suggests log or power.
    # Let's stick to linear prize for stability, but maybe square root to dampen outliers?
    # No, higher prize should be strictly better.
    
    base_heuristic = prizes[None, :] / safe_eff_dist
    
    # Apply feasibility: non-feasible edges get eps
    heuristic = np.where(feasible, base_heuristic, eps)
    
    # Slack boost: 
    # Edges with more slack are safer. 
    # However, too much slack might mean we are not utilizing the budget efficiently?
    # In orienteering, we want to maximize prize within max_len.
    # So we want to use the budget. 
    # A move that leaves 0 slack is risky (might not be feasible in practice if min_dist is not tight).
    # A move that leaves significant slack is safe.
    # Let's boost by (1 + slack / max_len) to favor safe moves slightly.
    # Or, we can penalize moves that are "too tight" (slack close to 0) more.
    
    # Let's try a different angle:
    # The probability of a move being feasible in the actual run depends on the actual path taken.
    # The heuristic is static.
    # A strong signal is: Can I still visit the best remaining nodes?
    # This is hard to compute statically for each edge.
    
    # Let's refine the base heuristic with a distance power.
    # If we use safe_eff_dist^1.2, we penalize long distances more.
    # Let's try exponent 1.0 for distance and 1.0 for prize.
    # The key improvement over the current program is using (d[i,j] + d[j,0]) in the denominator.
    
    # Let's also consider that moving to a node j with high d[j,0] is bad.
    # The current formula does this.
    
    # Let's add a small bonus for nodes that are "central" or have good connectivity?
    # No, keep it simple.
    
    # One more factor: The prize at node 0 is 0.
    # The heuristic for j=0 should be low.
    # prizes[0] = 0, so base_heuristic[:, 0] = 0.
    # This is correct.
    
    # Ensure all entries are positive and finite
    heuristic = np.where(heuristic > 0, heuristic, eps)
    heuristic = np.where(np.isfinite(heuristic), heuristic, eps)
    
    # Set diagonal to eps (self-loops not allowed/visited)
    np.fill_diagonal(heuristic, eps)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - start_time:
        pass
        
    return heuristic
# EVOLVE-BLOCK-END
