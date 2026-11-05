"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _select_heuristic(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic: Minimize dist(current, node) + dist(node, depot).
    This balances the cost of moving to the node with the cost of eventually returning to the depot.
    """
    if len(unvisited_nodes) == 0:
        return depot
    
    # Find feasible nodes
    feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    
    if len(feasible) == 0:
        return depot
    
    if len(feasible) == 1:
        return int(feasible[0])
    
    # Vectorized scoring
    # Score = dist(current, feasible) + dist(feasible, depot)
    d_from_current = distance_matrix[current_node, feasible]
    d_to_depot = distance_matrix[feasible, depot]
    
    # Weighted sum. 
    # A weight of 1.0 means we care equally about getting to the node and being close to the depot.
    # This encourages "compact" routes.
    scores = d_from_current + d_to_depot
    
    # To avoid getting stuck in local minima or bad capacity fills, 
    # we can check the top 3 candidates with a small lookahead.
    # If time permits.
    
    if _time_remaining() < 0.5:
        idx = np.argmin(scores)
        return int(feasible[idx])
    
    # Get top 3 candidates
    k = min(3, len(feasible))
    top_k_indices = np.argsort(scores)[:k]
    candidates = [feasible[i] for i in top_k_indices]
    
    best_node = candidates[0]
    best_cost = float('inf')
    
    # Lookahead depth 2
    for cand in candidates:
        sim_current = cand
        sim_remaining = rest_capacity - demands[cand]
        sim_cost = distance_matrix[current_node, cand] + distance_matrix[cand, depot] # Base score
        
        # Try to extend the route by 2 more steps greedily
        sim_unvisited = [n for n in unvisited_nodes if n != cand]
        
        for _ in range(2):
            if not sim_unvisited:
                break
            # Filter feasible
            sim_feasible = [n for n in sim_unvisited if demands[n] <= sim_remaining]
            if not sim_feasible:
                break
            
            # Pick best next based on same heuristic
            d_fc = distance_matrix[sim_current, sim_feasible]
            d_fd = distance_matrix[sim_feasible, depot]
            next_scores = d_fc + d_fd
            next_node = sim_feasible[np.argmin(next_scores)]
            
            sim_cost += distance_matrix[sim_current, next_node] + distance_matrix[next_node, depot] - distance_matrix[sim_current, depot]
            # Note: The above update is an approximation of the incremental cost.
            # Actually, simpler: just accumulate path length and add return at end.
            # Let's restart the cost calculation for clarity in lookahead.
            
        # Recalculate cost properly for the candidate path
        path_nodes = [cand]
        cur = cand
        rem = rest_capacity - demands[cand]
        unvisited_copy = [n for n in unvisited_nodes if n != cand]
        
        for _ in range(2):
            if not unvisited_copy:
                break
            f = [n for n in unvisited_copy if demands[n] <= rem]
            if not f:
                break
            d_fc = distance_matrix[cur, f]
            d_fd = distance_matrix[f, depot]
            sc = d_fc + d_fd
            nn = f[np.argmin(sc)]
            path_nodes.append(nn)
            rem -= demands[nn]
            cur = nn
            unvisited_copy.remove(nn)
            
        # Total cost of this simulated segment + return
        total_dist = distance_matrix[current_node, cand]
        for i in range(len(path_nodes) - 1):
            total_dist += distance_matrix[path_nodes[i], path_nodes[i+1]]
        total_dist += distance_matrix[path_nodes[-1], depot]
        
        if total_dist < best_cost:
            best_cost = total_dist
            best_node = cand
            
    return int(best_node)

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Main entry point.
    """
    if _time_remaining() < 0.1:
        # Fallback to fast nearest feasible
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return depot
        if len(feasible) == 1:
            return int(feasible[0])
        d = distance_matrix[current_node, feasible]
        return int(feasible[np.argmin(d)])

    if len(unvisited_nodes) == 0:
        return depot

    return _select_heuristic(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix)

# EVOLVE-BLOCK-END
