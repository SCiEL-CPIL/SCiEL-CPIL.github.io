import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # Safety margin before 150s hard kill

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node in the TSP tour using a robust heuristic with 2-opt local search.
    """
    if len(unvisited_nodes) == 0:
        return -1

    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])

    # Check time budget
    if time.time() - _START_TIME > _DEADLINE:
        # Fallback: nearest neighbor
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])

    K = len(unvisited_nodes)
    
    # Precompute distances from current to all unvisited
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Precompute distances from unvisited to destination
    dest_costs = distance_matrix[unvisited_nodes, destination_node]
    
    # Precompute distance matrix among unvisited nodes
    # unvisited_nodes is an array of indices
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    best_score = float('inf')
    best_node = int(unvisited_nodes[0])
    
    # Evaluate each candidate with a simplified 2-opt improvement on the remaining path
    for i, c in enumerate(unvisited_nodes):
        # Quick lower bound check
        # If d(current, c) + min(dist(c, unvisited\{c}) + dest_cost) is already worse, skip?
        # For now, just do the 2-opt on the hypothetical path.
        
        # Construct the hypothetical path:
        # current -> c -> [remaining unvisited in some order] -> destination
        # The "remaining unvisited" are unvisited_nodes excluding c.
        # We want to optimize the order of the remaining nodes starting from c and ending at destination.
        # Since we don't know the final order, we can approximate by:
        # 1. Greedy construction of the remaining path from c to destination.
        # 2. Apply 2-opt to improve it.
        
        # Greedy construction from c
        remaining = np.setdiff1d(unvisited_nodes, [c])
        if len(remaining) == 0:
            path_cost = dists_from_current[i] + dest_costs[i]
        else:
            # Build path: c -> r1 -> r2 -> ... -> rk -> destination
            # Greedy: at each step, pick nearest neighbor
            current_pos = c
            path_nodes = [c]
            visited_local = np.zeros(len(unvisited_nodes), dtype=bool)
            visited_local[i] = True # c is visited
            
            # We need to map unvisited_nodes indices to local indices for sub_matrix
            # But it's easier to work with global indices for distance_matrix
            
            while len(path_nodes) < K:
                # Find nearest unvisited node to current_pos
                # Candidates are unvisited_nodes where visited_local is False
                candidate_indices = np.where(~visited_local)[0]
                if len(candidate_indices) == 0:
                    break
                
                # Get global indices of candidates
                cand_globals = unvisited_nodes[candidate_indices]
                dists = distance_matrix[current_pos, cand_globals]
                min_idx_local = candidate_indices[np.argmin(dists)]
                next_node_global = unvisited_nodes[min_idx_local]
                
                path_nodes.append(next_node_global)
                visited_local[min_idx_local] = True
                current_pos = next_node_global
            
            # Calculate path cost
            path_cost = dists_from_current[i] # current -> c
            for j in range(len(path_nodes) - 1):
                path_cost += distance_matrix[path_nodes[j], path_nodes[j+1]]
            path_cost += distance_matrix[path_nodes[-1], destination_node]
            
            # Apply 2-opt to the sub-path (excluding current and destination for simplicity, or including?)
            # Let's apply 2-opt to the sequence of nodes in path_nodes
            # The path is: current -> path_nodes[0] -> ... -> path_nodes[-1] -> destination
            # We can optimize the order of path_nodes.
            
            improved_path = _two_opt_improve(path_nodes, current_node, destination_node, distance_matrix)
            
            # Recalculate cost with improved path
            improved_cost = distance_matrix[current_node, improved_path[0]]
            for j in range(len(improved_path) - 1):
                improved_cost += distance_matrix[improved_path[j], improved_path[j+1]]
            improved_cost += distance_matrix[improved_path[-1], destination_node]
            
            path_cost = min(path_cost, improved_cost)
        
        if path_cost < best_score:
            best_score = path_cost
            best_node = int(c)

    return best_node

def _two_opt_improve(path, start_node, end_node, distance_matrix):
    """
    Applies 2-opt improvement to a path.
    path: list of node indices
    start_node: node before path[0]
    end_node: node after path[-1]
    Returns improved path.
    """
    if len(path) < 3:
        return path
    
    improved = True
    path = list(path)
    n = len(path)
    
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Calculate delta cost for reversing path[i..j]
                # Current edges: (start, path[0]) ... (path[i-1], path[i]) ... (path[j], path[j+1]) ... (path[n-1], end)
                # New edges: (start, path[0]) ... (path[i-1], path[j]) ... (path[i], path[i+1]) ... (path[n-1], end)
                
                # Handle boundaries
                node_before_i = start_node if i == 0 else path[i-1]
                node_after_j = end_node if j == n-1 else path[j+1]
                
                # Remove edges: (node_before_i, path[i]) and (path[j], node_after_j)
                # Add edges: (node_before_i, path[j]) and (path[i], node_after_j)
                
                # Note: If i=0, the edge removed is (start, path[0])? No.
                # The path is start -> p0 -> p1 -> ... -> pn-1 -> end
                # 2-opt on p[i..j] reverses the segment.
                # The edges crossing the cut are:
                # 1. Edge connecting node before segment to first node of segment.
                # 2. Edge connecting last node of segment to node after segment.
                
                # If i == 0:
                #   Edge 1 is (start, path[0])
                # Else:
                #   Edge 1 is (path[i-1], path[i])
                
                # If j == n-1:
                #   Edge 2 is (path[j], end)
                # Else:
                #   Edge 2 is (path[j], path[j+1])
                
                if i == 0:
                    old_d1 = distance_matrix[start_node, path[0]]
                    new_d1 = distance_matrix[start_node, path[j]]
                else:
                    old_d1 = distance_matrix[path[i-1], path[i]]
                    new_d1 = distance_matrix[path[i-1], path[j]]
                
                if j == n - 1:
                    old_d2 = distance_matrix[path[j], end_node]
                    new_d2 = distance_matrix[path[i], end_node]
                else:
                    old_d2 = distance_matrix[path[j], path[j+1]]
                    new_d2 = distance_matrix[path[i], path[j+1]]
                
                delta = (new_d1 + new_d2) - (old_d1 + old_d2)
                
                if delta < -1e-8:
                    # Reverse the segment path[i..j]
                    path[i:j+1] = path[i:j+1][::-1]
                    improved = True
                    break
            if improved:
                break
                
    return path
