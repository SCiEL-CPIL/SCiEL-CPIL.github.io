"""Capacitated vehicle routing heuristic: route-level surrogate with angular clustering."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_DEADLINE = None
_START_TIME = None


def _ensure_deadline():
    global _DEADLINE, _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        _DEADLINE = _START_TIME + 115.0  # 5s safety margin before 120s
    return time.time() < _DEADLINE


def _simulate_nn_completion(
    current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix
):
    """
    Simulate nearest-neighbor completion of the route from current_node,
    starting with the given unvisited set and remaining capacity.
    Returns the predicted total additional route length (from current_node
    through remaining customers and back to depot).
    """
    if len(unvisited_nodes) == 0:
        return distance_matrix[current_node, depot]
    
    total_dist = 0.0
    current = current_node
    remaining_cap = rest_capacity
    unvisited = list(unvisited_nodes)
    
    while len(unvisited) > 0:
        # Find nearest feasible customer
        best_idx = -1
        best_dist = float('inf')
        for i, node in enumerate(unvisited):
            if demands[node] <= remaining_cap:
                d = distance_matrix[current, node]
                if d < best_dist:
                    best_dist = d
                    best_idx = i
        
        if best_idx == -1:
            # No feasible customer left; go back to depot
            total_dist += distance_matrix[current, depot]
            break
        
        node = unvisited[best_idx]
        total_dist += best_dist
        current = node
        remaining_cap -= demands[node]
        unvisited.pop(best_idx)
    
    # Return to depot
    total_dist += distance_matrix[current, depot]
    return total_dist


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Pick the next customer or depot (0) to close a route.
    Uses a route-level surrogate: for each feasible candidate, simulate NN
    completion and pick the one minimizing predicted total route length.
    Falls back to nearest feasible if time is tight or no feasible candidates.
    """
    if not _ensure_deadline():
        # Emergency fallback: nearest feasible
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return depot
        return int(feasible[np.argmin(distance_matrix[current_node, feasible])])
    
    unvisited_arr = np.asarray(unvisited_nodes)
    
    # Filter to feasible customers
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    feasible_nodes = unvisited_arr[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot
    
    # If only one feasible, take it
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])
    
    # For small sets, evaluate all candidates with route-level surrogate
    # For larger sets, use a hybrid: angular clustering + distance
    n_feasible = len(feasible_nodes)
    
    if n_feasible <= 15:
        # Full route-level surrogate evaluation
        best_node = int(feasible_nodes[0])
        best_score = float('inf')
        
        for node in feasible_nodes:
            # After visiting this node, what's the predicted completion?
            remaining_after = list(feasible_nodes)
            remaining_after.remove(int(node))
            
            # Distance to this node + simulated completion from this node
            d_to_node = distance_matrix[current_node, int(node)]
            new_cap = rest_capacity - demands[int(node)]
            
            if len(remaining_after) == 0:
                score = d_to_node + distance_matrix[int(node), depot]
            else:
                # Simulate NN from this node with remaining customers
                remaining_arr = np.asarray(remaining_after)
                sim_cap = new_cap
                sim_current = int(node)
                sim_dist = 0.0
                sim_remaining = list(remaining_arr)
                
                while len(sim_remaining) > 0:
                    if not _ensure_deadline():
                        break
                    # Nearest feasible
                    best_i = -1
                    best_d = float('inf')
                    for i, sn in enumerate(sim_remaining):
                        if demands[sn] <= sim_cap:
                            sd = distance_matrix[sim_current, sn]
                            if sd < best_d:
                                best_d = sd
                                best_i = i
                    if best_i == -1:
                        break
                    sn = sim_remaining[best_i]
                    sim_dist += best_d
                    sim_current = sn
                    sim_cap -= demands[sn]
                    sim_remaining.pop(best_i)
                
                sim_dist += distance_matrix[sim_current, depot]
                score = d_to_node + sim_dist
            
            if score < best_score:
                best_score = score
                best_node = int(node)
        
        return best_node
    else:
        # Larger set: use angular clustering + distance heuristic
        # Compute angles from depot to each feasible node
        # and from current node to each feasible node
        depot_coords = np.zeros(2)  # We don't have coordinates, use distance-based proxy
        
        # Use distance-based angular proxy:
        # For each candidate, compute a "direction score" based on
        # how well it continues the current direction (from depot to current)
        # and proximity to other feasible nodes (clustering)
        
        # Direction from depot to current
        d_depot_current = distance_matrix[depot, current_node]
        
        # For each candidate, score = distance from current + clustering penalty
        # Clustering: prefer nodes that are close to other feasible nodes (dense regions)
        
        best_node = int(feasible_nodes[0])
        best_score = float('inf')
        
        # Limit candidates for speed
        # Sort by distance from current, take top K
        dists = distance_matrix[current_node, feasible_nodes]
        sorted_idx = np.argsort(dists)
        top_k = min(15, n_feasible)
        candidates = feasible_nodes[sorted_idx[:top_k]]
        
        for node in candidates:
            node_int = int(node)
            d_to_node = distance_matrix[current_node, node_int]
            
            # Clustering bonus: average distance from this node to other feasible nodes
            # (lower = more clustered)
            if n_feasible > 1:
                other_dists = distance_matrix[node_int, feasible_nodes]
                # Exclude self
                avg_cluster = np.mean(other_dists)
            else:
                avg_cluster = 0.0
            
            # Depot closure cost if this were the last
            d_to_depot = distance_matrix[node_int, depot]
            
            # Composite score
            score = d_to_node + 0.3 * avg_cluster + 0.1 * d_to_depot
            
            if score < best_score:
                best_score = score
                best_node = node_int
        
        return best_node
# EVOLVE-BLOCK-END
