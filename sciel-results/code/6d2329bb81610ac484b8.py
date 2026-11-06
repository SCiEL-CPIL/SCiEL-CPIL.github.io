"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # 5s safety margin before the 120s kill

def _time_exceeded():
    return time.time() > _DEADLINE

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic: Capacity-aware Greedy 2-Opt Insertion Construction with 
    2-Opt and Or-opt Optimization and Perturbation.
    
    1. Construct an initial route using Greedy 2-Opt Insertion.
    2. Optimize using 2-Opt and efficient Or-opt.
    3. Try a simple perturbation (swap first two customers) and re-optimize.
    4. Return the first customer of the best optimized route.
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
    
    # 1. Construct initial route using Greedy 2-Opt Insertion
    # Start with the nearest neighbor
    remaining_cap = rest_capacity
    unvisited_in_seg = set(feasible)
    
    # Find initial nearest
    u_arr = np.array(list(unvisited_in_seg))
    dists = dm[current_node, u_arr]
    valid_mask = demands[u_arr] <= remaining_cap
    if not np.any(valid_mask):
        return depot
    valid_u = u_arr[valid_mask]
    valid_dists = dists[valid_mask]
    min_idx = np.argmin(valid_dists)
    first_node = valid_u[min_idx]
    
    route = [current_node, first_node]
    remaining_cap -= demands[first_node]
    unvisited_in_seg.remove(first_node)
    
    # Greedy Insertion: For each remaining node, pick the "nearest" one to the current end,
    # but insert it at the best position in the current route.
    # To keep it fast, we select the candidate node using NN from the last node, 
    # then find the best insertion point.
    
    while len(unvisited_in_seg) > 0:
        if _time_exceeded():
            break
            
        last_node = route[-1]
        u_arr = np.array(list(unvisited_in_seg))
        dists = dm[last_node, u_arr]
        
        # Filter by demand
        valid_mask = demands[u_arr] <= remaining_cap
        if not np.any(valid_mask):
            break
            
        valid_u = u_arr[valid_mask]
        valid_dists = dists[valid_mask]
        
        # Select the nearest feasible node to the current end
        min_idx = np.argmin(valid_dists)
        cand_node = valid_u[min_idx]
        
        # Find the best position to insert cand_node into route
        # route is [current, n1, n2, ..., nk]
        # We can insert at index 1..len(route)
        # Cost of route is sum(d[route[i], route[i+1]]) + d[route[-1], depot]
        
        best_pos = -1
        best_cost = float('inf')
        
        # Calculate current cost of route without the return to depot? 
        # Actually, we just need to compare the delta.
        # Let's calculate the full cost of the current route for reference? 
        # No, we can just calculate the cost of inserting at each position.
        
        # Current edges: (r[i], r[i+1]) for i=0..len-2, and (r[-1], depot)
        
        for pos in range(1, len(route) + 1):
            # Insert cand_node at position pos
            # This breaks edge (route[pos-1], route[pos]) if pos < len(route)
            # Or breaks (route[-1], depot) if pos == len(route)
            
            left = route[pos-1]
            if pos < len(route):
                right = route[pos]
                # Old edge: (left, right)
                # New edges: (left, cand), (cand, right)
                delta = dm[left, cand_node] + dm[cand_node, right] - dm[left, right]
            else:
                # Insert at the end
                # Old edge: (route[-1], depot)
                # New edges: (route[-1], cand), (cand, depot)
                delta = dm[left, cand_node] + dm[cand_node, depot] - dm[left, depot]
            
            # We want to minimize the total cost. Since the base cost is the same, 
            # minimizing delta is sufficient.
            if delta < best_cost:
                best_cost = delta
                best_pos = pos
        
        if best_pos != -1:
            route.insert(best_pos, cand_node)
            remaining_cap -= demands[cand_node]
            unvisited_in_seg.remove(cand_node)
        else:
            break

    if len(route) < 3:
        if len(route) > 1:
            return int(route[1])
        return depot

    # Helper to calculate route cost
    def get_route_cost(r):
        if len(r) <= 1:
            return 0.0
        cost = 0.0
        for i in range(len(r) - 1):
            cost += dm[r[i], r[i+1]]
        cost += dm[r[-1], depot]
        return cost

    best_route = route[:]
    best_cost = get_route_cost(best_route)

    # 2. Optimization Function
    def optimize_route(r):
        # 2-Opt
        improved = True
        max_iter = 15
        iter_count = 0
        n_route = len(r)
        
        while improved and iter_count < max_iter and not _time_exceeded():
            improved = False
            iter_count += 1
            
            for i in range(n_route - 2):
                if _time_exceeded():
                    return r
                    
                node_i = r[i]
                node_i1 = r[i+1]
                
                for j in range(i + 2, n_route):
                    node_j = r[j]
                    
                    if j < n_route - 1:
                        node_j1 = r[j+1]
                        old_len = dm[node_i, node_i1] + dm[node_j, node_j1]
                        new_len = dm[node_i, node_j] + dm[node_i1, node_j1]
                    else:
                        # j is the last node, edge is (node_j, depot)
                        old_len = dm[node_i, node_i1] + dm[node_j, depot]
                        new_len = dm[node_i, node_j] + dm[node_i1, depot]
                        
                    if new_len < old_len - 1e-10:
                        r[i+1:j+1] = reversed(r[i+1:j+1])
                        improved = True
                        if _time_exceeded():
                            return r

        # Or-opt (Single Node Move)
        improved = True
        max_iter = 8
        iter_count = 0
        
        while improved and iter_count < max_iter and not _time_exceeded():
            improved = False
            iter_count += 1
            n_route = len(r)
            
            for k in range(1, n_route): # Don't move the first node (current_node)
                if _time_exceeded():
                    return r
                    
                node_k = r[k]
                
                # Try inserting node_k at every other position m
                # We simulate the cost change
                # Removing node_k affects:
                # (r[k-1], r[k]) and (r[k], r[k+1] or depot)
                
                L = r[k-1]
                if k < n_route - 1:
                    R = r[k+1]
                else:
                    R = depot
                
                cost_remove = dm[L, node_k] + dm[node_k, R]
                cost_connect = dm[L, R]
                
                # Try inserting at m
                # We iterate m from 1 to n_route-1
                # Note: if m == k, no change.
                # We need to be careful with indices when m > k
                
                for m in range(1, n_route):
                    if m == k:
                        continue
                    
                    # Determine neighbors in the route *without* k
                    # This is tricky. Let's just copy the list for simplicity 
                    # given the small size, but we can optimize if needed.
                    # Since N is small (< 50), copying is O(N).
                    
                    r_copy = r[:]
                    r_copy.pop(k)
                    
                    # Adjust m index if m > k
                    insert_idx = m
                    if m > k:
                        insert_idx = m - 1
                        
                    # Bounds check for insert_idx in r_copy
                    # r_copy length is n_route - 1
                    # Valid indices for insertion: 1 to len(r_copy)-1? 
                    # We don't insert at 0 (before current_node).
                    # We can insert at the end? 
                    # If insert_idx == len(r_copy), it means after the last node.
                    # In r_copy, the last node is at len-1. 
                    # The "depot" is conceptually after the last node.
                    
                    if insert_idx < 1:
                        continue
                        
                    if insert_idx > len(r_copy) - 1:
                        # Insert at the end (before depot)
                        L_ins = r_copy[-1]
                        R_ins = depot
                        cost_insert_old = dm[L_ins, depot]
                        cost_insert_new = dm[L_ins, node_k] + dm[node_k, depot]
                    else:
                        L_ins = r_copy[insert_idx - 1]
                        R_ins = r_copy[insert_idx]
                        cost_insert_old = dm[L_ins, R_ins]
                        cost_insert_new = dm[L_ins, node_k] + dm[node_k, R_ins]
                    
                    delta = (cost_connect - cost_remove) + (cost_insert_new - cost_insert_old)
                    
                    if delta < -1e-10:
                        r_copy.insert(insert_idx, node_k)
                        r = r_copy
                        improved = True
                        break # Restart outer loop for this k? Or break entirely?
                        # Breaking m loop, restarting k loop is safer to avoid index issues
                        # But we modified r, so we should restart the while loop to be safe
                        # Actually, breaking the m loop and continuing the k loop with the new r 
                        # might be okay if we just break the k loop too and restart while.
            
            if improved:
                break # Restart the while loop

        return r

    # Run optimization on the constructed route
    optimized_route = optimize_route(best_route[:])
    optimized_cost = get_route_cost(optimized_route)
    
    # 3. Perturbation: Swap first two customers
    if len(optimized_route) >= 3 and not _time_exceeded():
        perturbed_route = optimized_route[:]
        # Swap index 1 and 2
        perturbed_route[1], perturbed_route[2] = perturbed_route[2], perturbed_route[1]
        
        perturbed_optimized = optimize_route(perturbed_route)
        perturbed_cost = get_route_cost(perturbed_optimized)
        
        if perturbed_cost < optimized_cost:
            best_route = perturbed_optimized
            best_cost = perturbed_cost
    
    if _time_exceeded():
        if len(best_route) > 1:
            return int(best_route[1])
        return depot

    if len(best_route) > 1:
        return int(best_route[1])
    else:
        return depot

# EVOLVE-BLOCK-END
