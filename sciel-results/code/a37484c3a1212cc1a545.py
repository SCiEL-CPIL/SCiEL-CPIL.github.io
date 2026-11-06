"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_DEADLINE = time.time() + 115.0

def _time_exceeded():
    return time.time() > _DEADLINE

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic: Capacity-aware Nearest Neighbor with 2-Opt local search.
    
    Instead of greedily picking the best next node, we construct a full route
    segment from the current node using a capacity-aware nearest neighbor rule,
    then apply 2-opt to minimize the length of that segment. The first node of
    the optimized segment is returned as the next step.
    """
    if len(unvisited_nodes) == 0:
        return depot

    # Filter by capacity
    feasible = [i for i in unvisited_nodes if demands[i] <= rest_capacity]
    if len(feasible) == 0:
        return depot
    if len(feasible) == 1:
        return int(feasible[0])

    # Time guard: if we are running out of time, fall back to simple nearest
    if _time_exceeded():
        idx = np.argmin(distance_matrix[current_node, feasible])
        return int(feasible[idx])

    dm = distance_matrix
    n = len(feasible)
    feas_arr = np.array(feasible)
    
    # 1. Construct initial route using Capacity-Aware Nearest Neighbor
    #    We simulate building a route starting from current_node.
    #    We keep track of remaining capacity and unvisited nodes in this segment.
    
    # We only consider nodes that fit in the current rest_capacity.
    # Since we are returning the NEXT node, we are effectively planning the rest of this route.
    
    # To make it efficient, we limit the depth of the look-ahead construction if n is large,
    # but for 50 customers, a full construction of the remaining route is feasible.
    
    # However, we must be careful: the "route" we construct is the sequence of nodes
    # starting from current_node, visiting a subset of feasible nodes, until capacity is full
    # or no feasible nodes remain (that fit).
    
    # Let's simulate the route construction.
    remaining_cap = rest_capacity
    unvisited_in_seg = set(feasible)
    route = [current_node]
    
    # We stop adding nodes when we can't fit any more or we run out of time
    # Note: In a real CVRP, we stop when capacity is full. Here, we simulate the greedy fill.
    
    # Optimization: If n is very large, we might just pick the nearest.
    # But 50 is small enough for a simple loop.
    
    while len(unvisited_in_seg) > 0:
        # Find nearest neighbor that fits
        best_node = -1
        best_dist = np.inf
        
        # Vectorized nearest neighbor search
        # Get distances from current_node to all unvisited_in_seg
        # We need to filter by demand <= remaining_cap
        
        # Convert set to array for numpy ops
        u_arr = np.array(list(unvisited_in_seg))
        dists = dm[current_node, u_arr]
        
        # Filter by demand
        valid_mask = demands[u_arr] <= remaining_cap
        if not np.any(valid_mask):
            break
            
        valid_u = u_arr[valid_mask]
        valid_dists = dists[valid_mask]
        
        min_idx = np.argmin(valid_dists)
        best_node = valid_u[min_idx]
        best_dist = valid_dists[min_idx]
        
        if best_node == -1:
            break
            
        # Add to route
        route.append(best_node)
        remaining_cap -= demands[best_node]
        unvisited_in_seg.remove(best_node)
        
        # Move current to best_node
        current_node = best_node
        
        # Time check inside loop
        if _time_exceeded():
            # Return the first node of the constructed route (which is the next step)
            # The route is [original_current, next, ...]
            # So the next node is route[1]
            if len(route) > 1:
                return int(route[1])
            else:
                return depot

    # 2. Apply 2-Opt to the constructed route segment
    # The route is: current_node -> node1 -> node2 -> ... -> nodek
    # We want to minimize the sum of distances between consecutive nodes.
    # Note: The return to depot is not included in this segment optimization 
    # because the next route will handle the depot return, or we assume the segment
    # ends when capacity is full and we return to depot. 
    # However, the cost of the segment includes the path from current_node to node1, etc.
    # The final return to depot is a fixed cost for this route if we close it here, 
    # but since we are only picking the NEXT node, we are committing to this sequence.
    # 2-Opt on the path (not cycle) is appropriate.
    
    if len(route) < 3:
        if len(route) > 1:
            return int(route[1])
        return depot

    # 2-Opt Implementation
    # route is a list of nodes.
    # We iterate over pairs (i, j) with 0 <= i < j < len(route)
    # If reversing route[i+1:j+1] reduces the total length, do it.
    
    # To be efficient, we limit the number of iterations or use a simple improvement loop.
    improved = True
    max_iter = 10 # Limit iterations to avoid timeout
    iter_count = 0
    
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        
        # Calculate current total length
        # L = sum(dm[route[k], route[k+1]] for k in range(len(route)-1))
        
        n_route = len(route)
        # We can precompute the edge lengths for faster update, but for small n, direct calc is fine.
        
        for i in range(n_route - 2):
            # i is the start of the segment to reverse
            # The edges affected are:
            # (route[i-1], route[i]) and (route[i], route[i+1]) if i>0
            # (route[j], route[j+1]) and (route[j+1], route[j+2]) if j < n_route-1
            
            # Standard 2-Opt for path:
            # Consider reversing route[i+1] to route[j]
            # Edges before: (route[i], route[i+1]) and (route[j], route[j+1])
            # Edges after:  (route[i], route[j]) and (route[i+1], route[j+1])
            
            for j in range(i + 2, n_route):
                # Indices in route list
                # Node at i-1 is route[i-1] (if i>0)
                # Node at i is route[i]
                # Node at i+1 is route[i+1]
                # Node at j is route[j]
                # Node at j+1 is route[j+1] (if j < n_route-1)
                
                # We only reverse the subarray route[i+1 : j+1]
                
                # Get the four nodes involved in the edge swap
                # Left side: route[i] connects to route[i+1]
                # Right side: route[j] connects to route[j+1] (if exists)
                
                # Actually, let's just calculate the cost difference.
                # Old edges: (route[i], route[i+1]) and (route[j], route[j+1])
                # New edges: (route[i], route[j]) and (route[i+1], route[j+1])
                
                # Note: If j is the last index, route[j+1] doesn't exist.
                # In a path 2-opt, if we reverse to the end, the right edge is not replaced, 
                # but the connection to the end changes.
                
                # Let's handle the general case where we reverse route[i+1:j+1].
                
                # Edge 1: route[i] -> route[i+1] becomes route[i] -> route[j]
                # Edge 2: route[j] -> route[j+1] becomes route[i+1] -> route[j+1]
                
                # If j is the last node (j == n_route - 1), there is no route[j+1].
                # In that case, the edge (route[j], route[j+1]) doesn't exist.
                # The new edge is (route[i+1], route[j])? No.
                
                # Let's trace:
                # Original: ... A - B - C - D ... E - F ...
                # i points to B, j points to F.
                # Reverse B..F: ... A - F - E - D - C - B ...
                # Old edges: (A,B) and (F, NextAfterF)
                # New edges: (A,F) and (B, NextAfterF)
                
                # So:
                # old_cost = dm[route[i], route[i+1]] + (dm[route[j], route[j+1]] if j < n_route-1 else 0)
                # new_cost = dm[route[i], route[j]] + (dm[route[i+1], route[j+1]] if j < n_route-1 else 0)
                
                # Wait, if j is the last node, the segment ends at j.
                # Original: ... A - B ... J
                # New: ... A - J ... B
                # Old edge: (A,B)
                # New edge: (A,J)
                # The edge after J doesn't exist.
                
                # Let's implement carefully.
                
                node_i = route[i]
                node_i1 = route[i+1]
                node_j = route[j]
                
                if j < n_route - 1:
                    node_j1 = route[j+1]
                    old_len = dm[node_i, node_i1] + dm[node_j, node_j1]
                    new_len = dm[node_i, node_j] + dm[node_i1, node_j1]
                else:
                    # j is the last node
                    old_len = dm[node_i, node_i1]
                    new_len = dm[node_i, node_j]
                    
                if new_len < old_len - 1e-10:
                    # Perform reversal
                    route[i+1:j+1] = reversed(route[i+1:j+1])
                    improved = True
                    if _time_exceeded():
                        break
            if _time_exceeded():
                break
        if _time_exceeded():
            break

    # The next node is the second element in the optimized route
    # route[0] is the original current_node
    # route[1] is the next customer
    if len(route) > 1:
        return int(route[1])
    else:
        return depot

# EVOLVE-BLOCK-END
