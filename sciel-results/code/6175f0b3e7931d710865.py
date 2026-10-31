# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: we must finish well within 150s wall-clock per evaluation.
# Each call to select_next_node is part of building one tour. The evaluator
# builds a full tour (50 steps) and scores it. We track cumulative time across
# calls within a single tour build.
_start_time = None
_DEADLINE = 110.0  # safety margin before 120s limit (evaluator kills at 150s, but let's be safe)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Pick the next city for a TSP constructive heuristic.
    
    Uses a 2-step lookahead with a soft look-ahead to balance immediate
    distance and future connectivity. Falls back to nearest neighbor
    as time budget is exhausted.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Check time budget
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5.0:
        # Emergency: just pick nearest neighbor
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
    
    # 2-step lookahead heuristic:
    # For each candidate, compute:
    #   score = dist(current, cand) + alpha * dist(cand, best_next(cand))
    # where best_next(cand) is the nearest unvisited node to cand (excluding cand itself)
    # and alpha balances short-term vs long-term cost.
    
    # When few nodes remain, also consider the cost to return to destination
    # (or start node, depending on how the tour is closed).
    
    candidates = unvisited_nodes
    n_candidates = n_unvisited
    
    # Precompute distances from current to all candidates (vectorized)
    dist_from_current = distance_matrix[current_node, candidates]  # shape (n_candidates,)
    
    # For each candidate, we need the nearest neighbor among other unvisited nodes
    # dist_matrix[cand, candidates] gives distances from cand to all candidates
    # We need to exclude self (the candidate itself)
    
    # Efficient approach: 
    # For each candidate i, find min of distance_matrix[candidates[i], candidates[j]] for j != i
    # This is O(n_candidates^2) total, which is 50^2 = 2500 operations - very fast.
    
    # Build the sub-matrix of distances between candidates
    sub_dist = distance_matrix[np.ix_(candidates, candidates)]  # shape (n_candidates, n_candidates)
    
    # For each row i, find the min excluding diagonal
    # Set diagonal to inf, then argmin
    diag_idx = np.arange(n_candidates)
    sub_dist_masked = sub_dist.copy()
    sub_dist_masked[diag_idx, diag_idx] = np.inf
    
    # Next-step cost for each candidate
    next_step_costs = np.min(sub_dist_masked, axis=1)  # shape (n_candidates,)
    
    # Alpha: balance immediate vs lookahead cost
    # As fewer nodes remain, the return-to-destination cost matters more
    # Use a decaying alpha: when many nodes remain, rely more on immediate distance
    # When few nodes remain, weight the lookahead/return cost more heavily
    
    # Also consider the cost to return to destination from the last node
    # For a constructive TSP, the final cost includes closing the tour
    # destination_node is the node we need to return to (typically the start)
    
    # When n_candidates is small (say <= 5), include the return cost explicitly
    if n_candidates <= 8:
        # dist_from_candidate_to_dest = distance_matrix[candidates, destination_node]
        # score = dist_from_current[i] + alpha * next_step_costs[i] + beta * dist_to_dest[i]
        # When only 1-2 nodes left, this becomes more precise
        
        dist_to_dest = distance_matrix[candidates, destination_node]
        
        # Weight: as we approach the end, the return cost dominates
        # alpha = 0.5, beta = 0.5 when 2 nodes left
        # alpha = 1.0, beta = 0.0 when many nodes left (but we only enter this branch for <=8)
        
        # Let's use a simple weighted combination
        # score = dist_from_current[i] + 0.5 * next_step_costs[i] + 0.5 * dist_to_dest[i]
        # But when n_candidates == 2, next_step_costs[i] is just the dist to the other candidate
        # and dist_to_dest[i] is the return cost. This is actually quite good.
        
        scores = dist_from_current + 0.5 * next_step_costs + 0.5 * dist_to_dest
    else:
        # For larger sets, use a lighter lookahead
        # alpha = 0.3 for moderate lookahead
        scores = dist_from_current + 0.3 * next_step_costs
    
    best_idx = int(np.argmin(scores))
    return int(candidates[best_idx])


# Reset the timer when a new tour build begins.
# We can detect this: if unvisited_nodes has all nodes (or is large), it's likely the start.
# But we don't know the total N. Alternative: reset if the first call has a large unvisited set.
# Actually, the evaluator calls this function for each step of one tour.
# We need to reset the timer per tour, not per call.
# Heuristic: if unvisited_nodes is large (e.g., > 40 for a 50-city instance), assume start of tour.

_original_select = select_next_node

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _start_time
    # Reset timer if this looks like the first call of a new tour
    # For a 50-city TSP, unvisited starts at 49 (after choosing start) or 50
    if len(unvisited_nodes) >= 45:
        _start_time = time.time()
    return _original_select(current_node, destination_node, unvisited_nodes, distance_matrix)
# EVOLVE-BLOCK-END
