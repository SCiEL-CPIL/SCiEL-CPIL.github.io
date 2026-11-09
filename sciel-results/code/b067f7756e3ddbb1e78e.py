import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Heuristic matrix for ant colony optimization of capacitated vehicle routing.
    
    Args:
        distance_matrix: (n, n) matrix of Euclidean distances
        coordinates: (n, 2) array of coordinates in unit square
        demands: (n,) array of demands (node 0 is depot with demand 0)
        capacity: vehicle capacity (50)
    
    Returns:
        (n, n) edge-attractiveness matrix where larger values are more attractive.
        Non-positive entries become 1e-9.
    """
    d = np.asarray(distance_matrix, dtype=float)
    coords = np.asarray(coordinates, dtype=float)
    dm = np.asarray(demands, dtype=float)
    n = d.shape[0]
    
    if d.shape[0] != d.shape[1]:
        return np.ones((n, n))
    
    # Time budget management
    deadline = time.time() + 270  # Safety margin before 330s limit
    
    # Base inverse distance heuristic with tuned exponent
    heuristic = np.zeros_like(d)
    mask = d > 1e-12
    heuristic[mask] = 1.0 / (d[mask] ** 2.5)
    
    # Demand factor: favor edges to high-demand nodes for vehicle filling
    demand_factor = 1.0 + dm / capacity
    heuristic = heuristic * demand_factor[np.newaxis, :]
    
    # Ensure positive values
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    # Clarke-Wright Savings with capacity constraint
    depot = 0
    customers = list(range(1, n))
    
    # Compute savings matrix
    savings = np.zeros((n, n))
    for i in range(1, n):
        for j in range(i + 1, n):
            s = d[depot, i] + d[depot, j] - d[i, j]
            savings[i, j] = s
            savings[j, i] = s
    
    # Sort savings in descending order
    pairs = [(savings[i, j], i, j) for i in range(1, n) for j in range(i + 1, n) if savings[i, j] > 0]
    pairs.sort(reverse=True)
    
    # Union-Find with capacity tracking
    parent = list(range(n))
    rank = [0] * n
    route_load = [0.0] * n
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        route_load[rx] = route_load[ry] + dm[rx] + dm[ry]
        return True
    
    # Initialize: each customer is its own route
    for i in range(1, n):
        route_load[i] = dm[i]
    
    # Process savings in order with capacity constraint - strict pass
    strict_edge_scores = np.zeros((n, n))
    for s, i, j in pairs:
        if time.time() > deadline:
            break
        
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        
        new_load = route_load[ri] + route_load[rj]
        if new_load > capacity:
            continue
        
        if union(ri, rj):
            # Score based on savings magnitude relative to max savings
            strict_edge_scores[i, j] = s
            strict_edge_scores[j, i] = s
    
    # Relaxed pass: allow higher capacity to identify more potential edges
    # This captures edges that might be useful in different route configurations
    if time.time() < deadline - 30:
        # Reset union-find for relaxed pass
        parent2 = list(range(n))
        rank2 = [0] * n
        route_load2 = [0.0] * n
        
        def find2(x):
            while parent2[x] != x:
                parent2[x] = parent2[parent2[parent2[x]]]
                x = parent2[x]
            return x
        
        def union2(x, y):
            rx, ry = find2(x), find2(y)
            if rx == ry:
                return False
            if rank2[rx] < rank2[ry]:
                rx, ry = ry, rx
            parent2[ry] = rx
            if rank2[rx] == rank2[ry]:
                rank2[rx] += 1
            route_load2[rx] = route_load2[ry] + dm[rx] + dm[ry]
            return True
        
        for i in range(1, n):
            route_load2[i] = dm[i]
        
        # Use 1.3x capacity for relaxed pass
        relaxed_capacity = capacity * 1.3
        relaxed_edge_scores = np.zeros((n, n))
        
        for s, i, j in pairs:
            if time.time() > deadline:
                break
            
            ri, rj = find2(i), find2(j)
            if ri == rj:
                continue
            
            new_load = route_load2[ri] + route_load2[rj]
            if new_load > relaxed_capacity:
                continue
            
            if union2(ri, rj):
                # Apply discount to relaxed scores since they're less certain
                relaxed_edge_scores[i, j] = s * 0.6
                relaxed_edge_scores[j, i] = s * 0.6
    
    # Combine strict and relaxed edge scores
    # Strict edges get higher weight, relaxed edges provide additional signal
    combined_scores = strict_edge_scores + relaxed_edge_scores
    
    # Convert combined scores to multiplicative heuristic boost
    # Use a smooth function: boost = 1 + alpha * (score / max_score)
    max_score = np.max(combined_scores)
    if max_score > 1e-12:
        normalized_scores = combined_scores / max_score
        # Boost factor: range from 1.0 (no edge) to 1 + 3.0 = 4.0 (best edge)
        boost = 1.0 + 3.0 * normalized_scores
    else:
        boost = np.ones((n, n))
    
    # Apply demand compatibility as a secondary smooth factor
    # Edges between nodes with complementary demands (sum close to capacity) are more valuable
    demand_compat = np.ones((n, n))
    for i in range(1, n):
        for j in range(i + 1, n):
            combined_demand = dm[i] + dm[j]
            if combined_demand <= capacity:
                # Fill ratio: how full the vehicle would be
                fill_ratio = combined_demand / capacity
                # Boost edges that lead to fuller loads (more efficient)
                # Use a smooth function that peaks around 0.8-0.9 fill
                compat_boost = 1.0 + 0.5 * fill_ratio * (1.0 - abs(fill_ratio - 0.85) / 0.85)
                demand_compat[i, j] = compat_boost
                demand_compat[j, i] = compat_boost
            else:
                # Penalize edges that would exceed capacity
                demand_compat[i, j] = 0.7
                demand_compat[j, i] = 0.7
    
    # Apply final heuristic
    heuristic = heuristic * boost * demand_compat
    
    # Ensure positive values and finite
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    return heuristic
