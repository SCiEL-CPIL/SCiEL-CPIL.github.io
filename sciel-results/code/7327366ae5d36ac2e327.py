"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > 145:
        return depot

    if len(unvisited_nodes) == 0:
        return depot

    # Convert to numpy arrays for vectorized operations
    unvisited_arr = np.array(unvisited_nodes, dtype=int)
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    feasible_nodes = unvisited_arr[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot

    # If only one feasible node, pick it
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])

    # Precompute distances
    d_depot_current = distance_matrix[depot, current_node]
    d_depot_cand = distance_matrix[depot, feasible_nodes]
    d_current_cand = distance_matrix[current_node, feasible_nodes]

    # Standard Clarke-Wright savings for linking current -> candidate
    savings = d_depot_current + d_depot_cand - d_current_cand

    # Capacity-aware penalty: as rest_capacity decreases, we should prefer nodes closer to depot
    # to avoid getting stuck far away with little capacity left.
    # Penalty increases as rest_capacity / total_capacity decreases.
    # Let's estimate total capacity or just use rest_capacity directly.
    # If rest_capacity is low, the weight on d_depot_cand should be higher.
    
    # A simple heuristic: weight = 1 + (1 - rest_capacity/50) * 2  (assuming max cap ~50)
    # Or better: weight based on how much demand is left relative to remaining capacity.
    # Let's use a dynamic weight that increases as rest_capacity drops.
    # If rest_capacity is 50, weight is 1.0. If 1, weight is high.
    weight = 1.0 + (1.0 - rest_capacity / 50.0) * 3.0
    
    # Adjusted score: We want high savings, but penalize far-from-depot nodes when capacity is tight.
    # Score = savings - weight * d_depot_cand
    # Wait, savings already includes d_depot_cand. 
    # savings = d_depot_current + d_depot_cand - d_current_cand
    # If we subtract weight * d_depot_cand, we get:
    # d_depot_current + (1-weight)*d_depot_cand - d_current_cand
    # If weight > 1, the coefficient of d_depot_cand becomes negative, effectively rewarding nodes CLOSER to depot.
    
    adjusted_savings = savings - weight * d_depot_cand

    # Lookahead Regret: 
    # Calculate the top 2 adjusted_savings values.
    # If the top 2 are close, the choice is less critical (low regret).
    # If the top is much higher than the second, picking the top is crucial (high regret if we don't).
    # However, we always pick the best. 
    # A different angle: "Regret" of NOT picking node i is the difference between the best score available 
    # and the score of node i. We want to pick the node that has the highest "opportunity cost" if skipped?
    # No, that's circular.
    
    # Let's try a different lookahead: 
    # For each candidate c, estimate the cost of the route if we pick c and then immediately close.
    # cost_close_c = d_current_cand[c] + d_depot_cand[c]
    # But we don't close immediately.
    
    # Let's stick to the improved savings with dynamic weight, but add a tie-breaker or secondary metric.
    # Secondary metric: Prefer nodes that allow more future flexibility?
    # Hard to compute without simulation.
    
    # Let's try: Score = adjusted_savings + lambda * (max_savings - savings)
    # This doesn't make sense.
    
    # Let's try a simple improvement: 
    # The current program picks argmax(savings).
    # My change: pick argmax(adjusted_savings) where adjusted_savings penalizes distance to depot more when capacity is low.
    
    idx = np.argmax(adjusted_savings)
    best_node = feasible_nodes[idx]
    
    return int(best_node)
# EVOLVE-BLOCK-END
