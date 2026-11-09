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
    d = np.array(distance_matrix, dtype=float)
    coords = np.array(coordinates, dtype=float)
    demands = np.array(demands, dtype=float)
    n = d.shape[0]
    
    # Ensure shape
    if d.shape[0] != d.shape[1]:
        return np.ones((n, n))
    
    # Start with base inverse distance heuristic
    heuristic = np.zeros_like(d)
    mask = d > 1e-12
    heuristic[mask] = 1.0 / (d[mask] ** 3)
    
    # Demand factor: favor edges to high-demand nodes
    demand_factor = 1.0 + demands / capacity
    heuristic = heuristic * demand_factor[np.newaxis, :]
    
    # Handle non-positive or non-finite values
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    # Run Clarke-Wright Savings to identify good edges
    # This provides a strong signal for the ACO
    deadline = time.time() + 280  # Safety margin before 330s limit
    
    # Compute savings: s(i,j) = d(0,i) + d(0,j) - d(i,j)
    depot = 0
    customers = list(range(1, n))
    
    # Calculate savings for all pairs of customers
    savings = np.zeros((n, n))
    for i in range(1, n):
        for j in range(i+1, n):
            s = d[depot, i] + d[depot, j] - d[i, j]
            savings[i, j] = s
            savings[j, i] = s
    
    # Sort savings in descending order
    pairs = [(savings[i, j], i, j) for i in range(1, n) for j in range(i+1, n) if savings[i, j] > 0]
    pairs.sort(reverse=True)
    
    # Implement Clarke-Wright Savings with capacity constraint
    # Track which nodes are in which route
    parent = list(range(n))
    rank = [0] * n
    route_load = [0.0] * n  # Load of each route (rooted at depot)
    
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
        route_load[rx] = route_load[ry] + demands[rx] + demands[ry]
        return True
    
    # Initialize: each customer is its own route
    for i in range(1, n):
        route_load[i] = demands[i]
    
    # Process savings in order
    good_edges = set()
    for s, i, j in pairs:
        if time.time() > deadline:
            break
        
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        
        # Check capacity constraint
        new_load = route_load[ri] + route_load[rj]
        if new_load > capacity:
            continue
        
        # Check that i and j are at the end of their respective routes
        # In a simplified version, we just check if they can be merged
        # For now, we'll accept the merge and track the edge
        if union(ri, rj):
            good_edges.add((min(i, j), max(i, j)))
    
    # Boost heuristic for good edges identified by Savings
    boost_factor = 5.0  # Strong boost for edges in the Savings solution
    
    for i, j in good_edges:
        heuristic[i, j] *= boost_factor
        heuristic[j, i] *= boost_factor
    
    # Final cleanup
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    return heuristic
