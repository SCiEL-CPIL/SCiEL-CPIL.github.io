"""Traveling salesman constructive heuristic with regret-based selection and fast NN completion."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state
_START_TIME = None
_TIME_LIMIT = 140.0  # Safety margin from 150s kill


def _ensure_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()


def _time_exceeded():
    global _START_TIME
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > _TIME_LIMIT


def _nearest_neighbor_completion_cost(
    start_node,
    end_node,
    unvisited_nodes,
    distance_matrix,
    max_steps=None,
):
    """
    Estimate the cost of completing the tour from start_node through all
    unvisited_nodes and ending at end_node using nearest-neighbor heuristic.
    
    Returns the estimated total cost.
    """
    if max_steps is None:
        max_steps = len(unvisited_nodes)
    
    n = len(unvisited_nodes)
    if n == 0:
        return distance_matrix[start_node, end_node]
    
    # We'll simulate NN greedily
    # To make it fast, we track which unvisited nodes have been "used"
    remaining = list(unvisited_nodes)
    total_cost = 0.0
    current = start_node
    steps = 0
    
    while len(remaining) > 0 and steps < max_steps:
        # Find nearest unvisited node
        dists = distance_matrix[current, remaining]
        min_idx = np.argmin(dists)
        total_cost += dists[min_idx]
        current = remaining[min_idx]
        remaining.pop(min_idx)
        steps += 1
    
    # Add cost to reach end_node
    total_cost += distance_matrix[current, end_node]
    
    return total_cost


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node in the TSP tour using a regret-based heuristic
    combined with nearest-neighbor completion cost estimation.
    
    The regret-2 heuristic picks the node where the gap between the best
    and second-best options is largest, which helps avoid "regret" from
    not picking a node that was uniquely good.
    
    We combine this with a fast NN completion cost estimate to look ahead.
    """
    _ensure_start_time()
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(current_node)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    if n_unvisited == 2:
        # Just pick the one that leads to shorter total path
        # Option 1: go to node 0, then node 1, then dest
        cost1 = distance_matrix[current_node, unvisited_nodes[0]] + \
                distance_matrix[unvisited_nodes[0], unvisited_nodes[1]] + \
                distance_matrix[unvisited_nodes[1], destination_node]
        # Option 2: go to node 1, then node 0, then dest
        cost2 = distance_matrix[current_node, unvisited_nodes[1]] + \
                distance_matrix[unvisited_nodes[1], unvisited_nodes[0]] + \
                distance_matrix[unvisited_nodes[0], destination_node]
        if cost1 <= cost2:
            return int(unvisited_nodes[0])
        else:
            return int(unvisited_nodes[1])
    
    # Time check - if we're running low, fall back to simple nearest neighbor
    if _time_exceeded():
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
    
    # Get distances from current node to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Get distances from destination node to all unvisited nodes
    dist_to_dest = distance_matrix[destination_node, unvisited_nodes]
    
    # Regret-2: for each candidate, compute the difference between
    # the best and second-best "total edge cost" involving that candidate.
    # Total edge cost for candidate i = dist(current, i) + dist(i, dest)
    # But we should really consider the full context.
    
    # A better regret: for each candidate i, consider the cost if we pick i now.
    # The "regret" is how much worse we'd be if we didn't pick i.
    # Approximation: regret(i) = min_j!=i [cost(j)] - cost(i)
    # where cost(i) = dist(current, i) + dist(i, dest)
    
    # But this is too simplistic. Let's use a lookahead approach.
    
    # Strategy: For each candidate, estimate the total remaining tour cost
    # using a fast NN completion from that candidate.
    # But this is O(n^2) per candidate, so O(n^3) total - too slow for n=50.
    
    # Instead, use a hybrid:
    # 1. Compute a base score for each candidate
    # 2. Use regret to break ties or adjust
    
    # Base score: weighted combination of distances
    # Emphasize distance from current, but also consider destination
    base_score = dist_from_current + 0.3 * dist_to_dest
    
    # For larger sets, add a connectivity penalty
    if n_unvisited > 5 and not _time_exceeded():
        # Estimate connectivity: for each candidate, how close is it to
        # other unvisited nodes on average?
        # This is O(n^2) which is fine for n<=50
        
        # Get submatrix of distances among unvisited nodes
        submatrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        # Set diagonal to inf to exclude self
        np.fill_diagonal(submatrix, np.inf)
        
        # Mean distance to other unvisited nodes
        mean_neighbor_dist = np.mean(submatrix, axis=1)
        
        # Add a small penalty for poorly connected nodes
        # Scale to be small relative to base_score
        penalty = 0.05 * mean_neighbor_dist
        base_score = base_score + penalty
    
    # Now apply regret-2: 
    # For each candidate i, regret = (second_best_score - best_score) where
    # we consider the scores of all candidates.
    # Actually, standard regret-2: for each candidate i, 
    # regret(i) = min_{j != i} score(j) - score(i)
    # Pick the candidate with maximum regret.
    
    # But this doesn't account for the fact that picking i changes the
    # problem for subsequent steps.
    
    # Let's try a different approach: use the base_score to rank candidates,
    # then for the top few candidates, do a more detailed evaluation.
    
    # Sort candidates by base_score
    sorted_indices = np.argsort(base_score)
    
    # Take top K candidates for detailed evaluation
    K = min(5, n_unvisited)
    top_candidates = sorted_indices[:K]
    
    # For each top candidate, estimate the remaining tour cost using
    # a fast NN completion
    best_candidate = top_candidates[0]
    best_estimated_cost = float('inf')
    
    for cand_idx in top_candidates:
        cand_node = unvisited_nodes[cand_idx]
        
        # Estimate cost: current -> cand -> (NN through remaining) -> dest
        # The cost of the first edge is dist_from_current[cand_idx]
        # Then we need to go from cand through remaining unvisited to dest
        
        # Fast NN completion from cand through remaining to dest
        remaining = list(unvisited_nodes)
        remaining.pop(cand_idx)  # Remove the candidate we're considering
        
        if len(remaining) == 0:
            est_cost = dist_from_current[cand_idx] + distance_matrix[cand_node, destination_node]
        else:
            # Quick NN estimate
            remaining_arr = np.array(remaining)
            # Simulate NN
            curr = cand_node
            total = dist_from_current[cand_idx]
            rem_list = list(remaining_arr)
            
            # Limit steps for speed
            max_steps = min(len(rem_list), 15)
            for _ in range(max_steps):
                if not rem_list:
                    break
                d = distance_matrix[curr, rem_list]
                min_i = np.argmin(d)
                total += d[min_i]
                curr = rem_list[min_i]
                rem_list.pop(min_i)
            
            if rem_list:
                # If we didn't finish, add a rough estimate
                # Average remaining distance
                remaining_arr2 = np.array(rem_list)
                avg_dist = np.mean(distance_matrix[curr, remaining_arr2])
                total += avg_dist * len(rem_list)
            
            total += distance_matrix[curr, destination_node]
            est_cost = total
        
        if est_cost < best_estimated_cost:
            best_estimated_cost = est_cost
            best_candidate = cand_idx
    
    return int(unvisited_nodes[best_candidate])
# EVOLVE-BLOCK-END
