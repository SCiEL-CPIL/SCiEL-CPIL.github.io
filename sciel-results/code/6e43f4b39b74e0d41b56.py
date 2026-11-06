import numpy as np
import time

# Global deadline guard
_DEADLINE = time.time() + 145  # Safety margin for 150s limit

def _lower_bound_estimate(current, unvisited, destination, dist_matrix):
    """
    Estimate the cost to complete the tour from 'current' visiting all 'unvisited' and returning to 'destination'.
    Uses a tighter lower bound:
    1. Min cost to enter the set of unvisited from current.
    2. Min cost to leave the set of unvisited to destination.
    3. Approximate MST cost on unvisited nodes (sum of 2 smallest edges per node / 2).
    """
    if len(unvisited) == 0:
        return dist_matrix[current, destination]
    
    n = len(unvisited)
    if n == 1:
        return dist_matrix[current, unvisited[0]] + dist_matrix[unvisited[0], destination]
    
    # Min edge from current to any unvisited
    min_in = np.min(dist_matrix[current, unvisited])
    # Min edge from any unvisited to destination
    min_out = np.min(dist_matrix[unvisited, destination])
    
    # Approximate MST cost for unvisited nodes
    sub_dist = dist_matrix[np.ix_(unvisited, unvisited)]
    np.fill_diagonal(sub_dist, np.inf)
    # Get 2 smallest per row
    mins = np.partition(sub_dist, 1, axis=1)[:, :2]
    mst_cost = np.sum(mins) / 2.0
    
    return min_in + min_out + mst_cost

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _DEADLINE
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Time check: if we're running low on time, fall back to simple nearest neighbor
    if time.time() > _DEADLINE - 1.0:
        dists = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists)
        return int(unvisited_nodes[best_idx])
    
    if n_unvisited == 2:
        a = int(unvisited_nodes[0])
        b = int(unvisited_nodes[1])
        cost_a = distance_matrix[current_node, a] + distance_matrix[a, b] + distance_matrix[b, destination_node]
        cost_b = distance_matrix[current_node, b] + distance_matrix[b, a] + distance_matrix[a, destination_node]
        return a if cost_a < cost_b else b
    
    # Beam Search Strategy
    # State: (path_nodes, cost_so_far, last_node)
    # We maintain a beam of the best 'width' paths of length 'depth'.
    
    # Tune beam width and depth based on problem size
    if n_unvisited <= 10:
        BEAM_WIDTH = 30
        DEPTH = 4
    elif n_unvisited <= 20:
        BEAM_WIDTH = 25
        DEPTH = 3
    else:
        BEAM_WIDTH = 20
        DEPTH = 3
    
    # Initial beam: one path starting at current_node
    beam = [(0.0, tuple(), current_node)]
    
    for step in range(DEPTH):
        if not beam:
            break
            
        next_beam = []
        
        for cost_so_far, path, last in beam:
            visited_set = set(path)
            
            for node in unvisited_nodes:
                if node in visited_set:
                    continue
                
                new_cost = cost_so_far + distance_matrix[last, node]
                new_path = path + (node,)
                next_beam.append((new_cost, new_path, node))
        
        if not next_beam:
            break
        
        # Sort by cost and keep top BEAM_WIDTH
        next_beam.sort(key=lambda x: x[0])
        beam = next_beam[:BEAM_WIDTH]
        
        # Early termination if time is tight
        if time.time() > _DEADLINE - 2.0:
            if beam and len(beam[0][1]) > 0:
                return int(beam[0][1][0])
            else:
                dists = distance_matrix[current_node, unvisited_nodes]
                best_idx = np.argmin(dists)
                return int(unvisited_nodes[best_idx])
    
    # After DEPTH steps, select the path with the lowest estimated total cost
    best_total_cost = float('inf')
    best_first_node = -1
    
    for cost_so_far, path, last in beam:
        if len(path) == 0:
            continue
            
        visited_set = set(path)
        remaining = np.array([n for n in unvisited_nodes if n not in visited_set])
        
        lb = _lower_bound_estimate(last, remaining, destination_node, distance_matrix)
        total_est = cost_so_far + lb
        
        if total_est < best_total_cost:
            best_total_cost = total_est
            best_first_node = int(path[0])
                
    if best_first_node == -1:
        # Fallback to nearest neighbor
        dists = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists)
        best_first_node = int(unvisited_nodes[best_idx])
        
    return best_first_node
