"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for deadline tracking
_start_time = None
_deadline = 145.0  # 5s buffer before 150s limit

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time
    
    # Initialize deadline tracking on first call
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: stop if we've exceeded deadline
    elapsed = time.time() - _start_time
    if elapsed > _deadline:
        return depot
    
    if len(unvisited_nodes) == 0:
        return depot
    
    # Check feasible nodes (demand <= rest_capacity)
    unvisited_arr = np.array(unvisited_nodes)
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    feasible_nodes = unvisited_arr[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot
    
    # If only one feasible node, take it
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])
    
    # Regret-based selection:
    # For each feasible node c, calculate:
    # 1. d_current_c: distance from current node to c
    # 2. d_depot_c: distance from depot to c (proxy for future route start/end)
    # 3. Min distance from c to any OTHER feasible node (excluding current, but we are moving from current)
    #    Actually, regret is usually: cost_if_not_chosen_now - cost_if_chosen_now
    #    In VRP context, if we don't visit c now, we might have to visit it later from a different node.
    #    
    # Let's use a simpler "regret" heuristic adapted for sequential construction:
    # Score(c) = d(current, c) + d(depot, c) - (min_{c' in feasible, c' != c} d(c, c'))
    # This penalizes nodes that are far from current OR far from depot, but rewards them if they are
    # close to other feasible nodes (meaning we can visit them in a cluster).
    # Wait, standard regret is: (2nd best option cost) - (best option cost).
    # Here, "cost" is the incremental distance.
    # If we pick c, cost is d(current, c).
    # If we don't pick c, what is the cost? It's harder to define sequentially.
    
    # Alternative: Lookahead 1-step regret.
    # For each candidate c, assume we pick c. Then what is the best next move from c?
    # Let next_best(c) = min_{c' in feasible, c' != c} d(c, c')
    # Total estimated cost for path current -> c -> next_best(c) is d(current, c) + next_best(c)
    # We want to minimize this.
    # However, we also need to consider returning to depot eventually.
    # Let's add d(depot, c) as a penalty for being far from depot (since we must return).
    
    # Actually, a more robust "regret" in this context is:
    # How much worse is it to visit c later than now?
    # Hard to compute exactly.
    
    # Let's try a hybrid:
    # Primary: Minimize d(current, c) + d(c, nearest_other_feasible)
    # This encourages visiting nodes that are close to current AND close to other nodes (clustering).
    # Secondary: If ties, prefer closer to depot.
    
    # Calculate distances from current to all feasible
    d_current_to_feasible = distance_matrix[current_node, feasible_nodes]
    
    # Calculate distances between feasible nodes
    # We need the minimum distance from each feasible node to any other feasible node
    # This is O(N^2) where N is number of feasible nodes. N <= 50. 2500 ops. Fast.
    
    min_dist_to_other = np.full(len(feasible_nodes), np.inf)
    
    # If N is small, loop is fine. If N is large, vectorize.
    # Since max 50 customers, N is small.
    for i in range(len(feasible_nodes)):
        ci = feasible_nodes[i]
        # Distances from ci to all other feasible nodes
        dists_to_others = distance_matrix[ci, feasible_nodes]
        # Set self-distance to inf
        dists_to_others[i] = np.inf
        if len(dists_to_others) > 1:
            min_dist_to_other[i] = np.min(dists_to_others)
        else:
            min_dist_to_other[i] = 0.0 # Should not happen due to len check above
    
    # Score: d(current, c) + min_dist_to_other
    # We want to minimize this score.
    scores = d_current_to_feasible + min_dist_to_other
    
    # Tie-breaker or refinement: Add a small penalty for distance to depot
    # This helps close routes when we are far from cluster but close to depot
    d_depot_to_feasible = distance_matrix[depot, feasible_nodes]
    
    # Weighted sum: 0.9 * (local cluster cost) + 0.1 * (depot distance)
    # The local cost encourages moving to the next node in a tight cluster.
    # The depot distance encourages not straying too far if we are running low on capacity or time.
    # Actually, let's stick to the pure lookahead cost for the step, but add depot distance
    # only if the node is the last feasible one? No, we don't know if it's last.
    
    # Let's refine: The "regret" of not visiting c now is that we might have to visit it from a worse position later.
    # A common heuristic is to prioritize nodes with high "urgency".
    # Urgency(c) = d(current, c) - min_{c' in feasible, c' != c} d(current, c')
    # If c is the closest to current, urgency is low (0). If c is far, urgency is high.
    # But we want to pick the BEST node, not the most urgent.
    
    # Let's go back to the "Lookahead 1-step" minimization.
    # Cost(c) = d(current, c) + min_{c' in feasible, c' != c} d(c, c')
    # This effectively plans a 2-step path.
    
    idx = np.argmin(scores)
    best_node = feasible_nodes[idx]
    
    return int(best_node)
# EVOLVE-BLOCK-END
