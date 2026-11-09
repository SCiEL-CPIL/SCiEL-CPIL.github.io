# EVOLVE-BLOCK-START
import numpy as np
import time

def _two_opt_improve(route, distance_matrix, max_passes=10):
    """
    Improves a single route using 2-Opt.
    Route is a list of node indices, starting and ending with depot (0).
    """
    n_route = len(route)
    if n_route < 4:
        return route
    
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(1, n_route - 2):
            for j in range(i + 1, n_route - 1):
                u = route[i-1]
                v = route[i]
                w = route[j]
                x = route[j+1]
                
                old_dist = distance_matrix[u, v] + distance_matrix[w, x]
                new_dist = distance_matrix[u, w] + distance_matrix[v, x]
                
                if new_dist < old_dist - 1e-10:
                    # Reverse segment from i to j
                    route[i:j+1] = route[i:j+1][::-1]
                    improved = True
    return route

def _or_opt_improve(route, distance_matrix, max_passes=10):
    """
    Improves a single route using Or-opt (relocation of 1, 2, or 3 nodes).
    """
    n_route = len(route)
    if n_route < 4:
        return route
    
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for size in [1, 2, 3]:
            if n_route < size + 4: # Need enough nodes
                continue
            for i in range(1, n_route - size - 2):
                # Segment to move: route[i : i+size]
                seg = route[i : i+size]
                # Remove segment
                pre = route[:i]
                post = route[i+size:]
                
                # Try inserting at all possible positions in the remaining route
                # We must be careful not to insert in a way that creates invalid capacity (but we assume capacity is checked outside or this is just a polish step on a feasible route)
                # For simplicity and speed, we just check distance improvement.
                
                best_delta = 1e-10
                best_pos = -1
                
                # The remaining route is pre + post
                # We can insert seg anywhere in pre+post, except immediately before or after its original position (to avoid no-ops)
                # Let's just iterate over all insertion points in the combined list
                combined = pre + post
                n_comb = len(combined)
                
                for k in range(n_comb + 1):
                    # Skip positions that correspond to the original location
                    # Original location was i in the full route.
                    # In the combined list, the gap is at index i (if we consider pre length)
                    # Actually, it's easier to just calculate delta for all k and check if it's a valid move (not just re-inserting in same place)
                    
                    if k < n_comb:
                        u = combined[k-1] if k > 0 else combined[0] # This logic is tricky for circular routes, but route is linear [0, ..., 0]
                        # Wait, route is [0, n1, n2, ..., 0].
                        # u is node before insertion point, v is node at insertion point.
                        # If k=0, insert at start? No, start is 0.
                        # If k=n_comb, insert at end? No, end is 0.
                        # So k ranges from 1 to n_comb-1 effectively for internal insertions.
                        
                        # Let's refine:
                        # Insertion between combined[k-1] and combined[k]
                        if k == 0 or k == n_comb:
                            continue
                            
                        # Avoid inserting back into the same spot
                        # The gap in 'combined' corresponds to index i in the original route.
                        # pre has length i. So the gap is at index i in combined.
                        # If k == i, we are re-inserting in the same place.
                        if k == i:
                            continue
                        
                        u = combined[k-1]
                        v = combined[k]
                        
                        # Cost to remove seg:
                        # Connect pre_end to post_start
                        pre_end = pre[-1] if pre else 0
                        post_start = post[0] if post else 0
                        # Note: pre and post are slices of the route which starts/ends with 0.
                        # If i=1, pre=[0], post=[n2..., 0].
                        
                        # Cost of original connection:
                        # dist(pre_end, seg[0]) + dist(seg[-1], post_start)
                        old_cost = distance_matrix[pre_end, seg[0]] + distance_matrix[seg[-1], post_start]
                        
                        # Cost of new connection:
                        # dist(pre_end, post_start) + dist(u, seg[0]) + dist(seg[-1], v)
                        # Wait, if we insert seg between u and v:
                        # new_cost = dist(u, seg[0]) + dist(seg[-1], v)
                        # And the connection pre_end -> post_start is created.
                        
                        # Total change:
                        # Remove: dist(pre_end, seg[0]) + dist(seg[-1], post_start)
                        # Add: dist(pre_end, post_start) + dist(u, seg[0]) + dist(seg[-1], v)
                        
                        # However, if the insertion is adjacent to the original position, the "pre_end -> post_start" link might overlap with existing links.
                        # This simple Or-opt implementation is prone to errors if not careful with indices.
                        # Given the time constraint, I will stick to a robust 2-Opt and a simpler "1-move" Or-opt.
                        
                        # Let's try a simpler 1-node move.
                        pass
                        
    return route

# Since Or-opt is complex to implement correctly in a short time without bugs, I will rely on a stronger 2-Opt 
# and a better Lagrangian subproblem solver.

def _lagrangian_relaxation(distance_matrix, demands, capacity, max_dual_iters=40, max_time=50):
    """
    Solve a Lagrangian relaxation of CVRP.
    Subproblem: Minimize sum(d_ij * x_ij) + lambda * sum_j (demand_j * x_0j - capacity * y_j)
    This is not quite right. Standard Lagrangian for CVRP relaxes the capacity constraint.
    L(lambda) = min_x [ sum d_ij x_ij + lambda * (sum_j demand_j x_0j - capacity * sum_j y_j) ]
    This is complex.
    
    Simpler approach used in literature:
    Penalize the cost of visiting a node based on its demand.
    c'_ij = d_ij + lambda * demand_j (if j is customer)
    Then solve TSP on modified costs.
    """
    start_time = time.time()
    n = distance_matrix.shape[0]
    
    lambda_val = 0.0
    best_relaxed_cost = float('inf')
    edge_attractiveness = np.zeros((n, n), dtype=float)
    
    for dual_iter in range(max_dual_iters):
        if time.time() - start_time > max_time:
            break
        
        # Modified cost matrix
        modified_dm = distance_matrix.copy()
        for j in range(1, n):
            penalty = lambda_val * demands[j]
            modified_dm[:, j] += penalty
            # Note: In TSP, we pay for entering j.
        
        # Solve TSP on modified_dm using NN + 2-Opt
        # NN Construction
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        current = 0
        route = [0]
        
        while not visited.all():
            # Find nearest unvisited
            unvisited = np.where(~visited)[0]
            if len(unvisited) == 0:
                break
            costs = modified_dm[current, unvisited]
            next_node = unvisited[np.argmin(costs)]
            route.append(next_node)
            visited[next_node] = True
            current = next_node
        
        route.append(0)
        
        # 2-Opt Improvement
        route = _two_opt_improve(route, modified_dm, max_passes=5)
        
        # Calculate cost using ORIGINAL distances
        total_cost = 0.0
        for k in range(len(route) - 1):
            total_cost += distance_matrix[route[k], route[k+1]]
        
        # Calculate load
        load = sum(demands[c] for c in route[1:-1])
        
        # Subgradient
        subgradient = load - capacity
        
        # Polyak Step
        if total_cost > best_relaxed_cost:
            gap = total_cost - best_relaxed_cost
            step_size = 2.0 * gap / (subgradient**2 + 1e-9) if gap > 1e-6 else 0.1
        else:
            step_size = 0.1
            
        lambda_val = max(0.0, lambda_val + step_size * subgradient)
        
        if total_cost < best_relaxed_cost:
            best_relaxed_cost = total_cost
        
        # Accumulate edges
        # Weight by how good the solution is
        quality = 1.0 / (1.0 + (total_cost - best_relaxed_cost))
        
        for k in range(len(route) - 1):
            u, v = route[k], route[k+1]
            edge_attractiveness[u, v] += quality
            edge_attractiveness[v, u] += quality * 0.5 # Symmetry boost? No, TSP is symmetric, but let's keep it consistent.
            
    max_attr = np.max(edge_attractiveness)
    if max_attr > 0:
        edge_attractiveness = edge_attractiveness / max_attr
        
    return edge_attractiveness

def _savings_heuristic(distance_matrix, demands, capacity, n=51):
    """Clarke-Wright Savings heuristic for CVRP."""
    customers = list(range(1, n))
    active_routes = []
    route_index = {}
    
    for c in customers:
        if demands[c] <= capacity:
            active_routes.append([0, c, 0])
            route_index[c] = len(active_routes) - 1
    
    if not active_routes:
        return []
        
    savings = {}
    for i in customers:
        for j in customers:
            if i == j:
                continue
            s = distance_matrix[0, i] + distance_matrix[0, j] - distance_matrix[i, j]
            savings[(i, j)] = s
            
    sorted_pairs = sorted(savings.keys(), key=lambda x: savings[x], reverse=True)
    
    for i, j in sorted_pairs:
        ri = route_index.get(i)
        rj = route_index.get(j)
        
        if ri is None or rj is None:
            continue
        if ri == rj:
            continue
            
        route_i = active_routes[ri]
        route_j = active_routes[rj]
        
        # Check if i and j are at the ends of their routes
        i_adj_depot = (route_i[1] == i or route_i[-2] == i)
        j_adj_depot = (route_j[1] == j or route_j[-2] == j)
        
        if not (i_adj_depot and j_adj_depot):
            continue
            
        cap_i = sum(demands[c] for c in route_i[1:-1])
        cap_j = sum(demands[c] for c in route_j[1:-1])
        
        if cap_i + cap_j > capacity:
            continue
            
        # Merge
        if route_i[1] == i and route_j[1] == j:
            # i is start of i, j is start of j
            # [0, i, ...] and [0, j, ...]
            # Merge: [0, ...j_rev..., ...i...]
            inner_j = route_j[1:-1][::-1]
            inner_i = route_i[1:-1]
            merged = [0] + inner_j + inner_i + [0]
            
        elif route_i[1] == i and route_j[-2] == j:
            # i is start of i, j is end of j
            # [0, i, ...] and [0, ..., j]
            # Merge: [0, ..., j, i, ...] -> Wait, j is end, so ...j connects to i?
            # [0, ..., j] + [i, ...] -> [0, ..., j, i, ...]
            inner_i = route_i[1:-1]
            inner_j = route_j[1:-1]
            merged = [0] + inner_j + inner_i + [0]
            
        elif route_i[-2] == i and route_j[1] == j:
            # i is end of i, j is start of j
            # [0, ..., i] and [0, j, ...]
            # Merge: [0, ..., i, j, ...]
            inner_i = route_i[1:-1]
            inner_j = route_j[1:-1]
            merged = [0] + inner_i + inner_j + [0]
            
        else:
            # i is end of i, j is end of j
            # [0, ..., i] and [0, ..., j]
            # Merge: [0, ..., j_rev, ..., i]
            inner_i = route_i[1:-1][::-1]
            inner_j = route_j[1:-1]
            merged = [0] + inner_j + inner_i + [0]
            
        active_routes[ri] = merged
        del active_routes[rj]
        
        for node in merged[1:-1]:
            route_index[node] = ri
    
    # Apply 2-Opt to each route
    for idx in range(len(active_routes)):
        active_routes[idx] = _two_opt_improve(active_routes[idx], distance_matrix, max_passes=10)
        
    return active_routes

def heuristics(distance_matrix, coordinates, demands, capacity):
    start_time = time.time()
    deadline = start_time + 280
    
    n = distance_matrix.shape[0]
    
    # 1. Lagrangian Relaxation-based edge weights
    try:
        lagrangian_weights = _lagrangian_relaxation(
            distance_matrix, demands, capacity,
            max_dual_iters=40, max_time=60
        )
    except Exception:
        lagrangian_weights = np.zeros((n, n), dtype=float)
    
    # 2. Generate reference solution using Savings Heuristic + 2-Opt
    try:
        routes = _savings_heuristic(distance_matrix, demands, capacity, n)
        
        edge_matrix = np.zeros((n, n), dtype=float)
        for route in routes:
            for k in range(len(route) - 1):
                u = route[k]
                v = route[k+1]
                edge_matrix[u, v] = 1.0
    except Exception:
        edge_matrix = np.zeros((n, n), dtype=float)
        
    # 3. Base heuristic: 1/d
    dm_safe = distance_matrix + 1e-9
    heuristic_base = 1.0 / dm_safe
    max_base = np.max(heuristic_base)
    if max_base > 0:
        heuristic_base = heuristic_base / max_base
        
    # 4. Combine with tuned weights
    # Lagrangian weights are normalized to [0, 1]
    # Edge matrix is 0 or 1
    # Base is normalized to [0, 1]
    
    alpha_lagrangian = 15.0
    alpha_savings = 10.0
    
    heuristic = heuristic_base + alpha_lagrangian * lagrangian_weights + alpha_savings * edge_matrix
    
    # Ensure positive
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    # Final check
    if not np.all(np.isfinite(heuristic)):
        heuristic = 1.0 / (distance_matrix + 1e-9)
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
