import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Heuristic matrix for ant colony optimization of capacitated vehicle routing.
    Uses a smooth surrogate via log-sum-exp to aggregate multi-start savings-based
    constructive heuristics into a single edge-attractiveness matrix.
    """
    d = np.asarray(distance_matrix, dtype=float)
    coords = np.asarray(coordinates, dtype=float)
    demands = np.asarray(demands, dtype=float)
    n = d.shape[0]
    
    # Ensure valid shape
    if d.shape[0] != d.shape[1]:
        return np.ones((n, n))
    
    # Base inverse distance heuristic
    heuristic = np.zeros_like(d)
    mask = d > 1e-12
    heuristic[mask] = 1.0 / (d[mask] ** 2.0)
    
    # Demand factor: favor edges to high-demand nodes
    demand_factor = 1.0 + demands / capacity
    heuristic = heuristic * demand_factor[np.newaxis, :]
    
    # Handle non-positive or non-finite values
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    deadline = time.time() + 280  # Safety margin before 330s limit
    
    # Compute savings: s(i,j) = d(0,i) + d(0,j) - d(i,j)
    depot = 0
    savings = np.zeros((n, n))
    for i in range(1, n):
        for j in range(i + 1, n):
            s = d[depot, i] + d[depot, j] - d[i, j]
            savings[i, j] = s
            savings[j, i] = s
    
    # Multi-start Clarke-Wright Savings with capacity constraint
    # Each run uses a different perturbation of the savings to get diverse solutions
    # We accumulate edge usage counts across multiple runs
    
    # Edge usage accumulator: (i, j) -> count
    edge_counts = np.zeros((n, n), dtype=np.float64)
    
    # Number of multi-start runs
    num_starts = 20
    
    # Pre-compute customer indices
    customers = list(range(1, n))
    
    for start in range(num_starts):
        if time.time() > deadline:
            break
        
        # Perturb savings with random noise for diversity
        # Use a small random perturbation scaled by the savings magnitude
        np.random.seed(start * 1000 + 42)
        noise_scale = 0.1 if start == 0 else 0.3
        perturbed_savings = savings.copy()
        for i in range(1, n):
            for j in range(i + 1, n):
                if perturbed_savings[i, j] > 0:
                    perturbed_savings[i, j] *= (1.0 + np.random.uniform(-noise_scale, noise_scale))
                    perturbed_savings[j, i] = perturbed_savings[i, j]
        
        # Sort pairs by perturbed savings in descending order
        pairs = []
        for i in range(1, n):
            for j in range(i + 1, n):
                if perturbed_savings[i, j] > 0:
                    pairs.append((perturbed_savings[i, j], i, j))
        pairs.sort(reverse=True)
        
        # Union-Find with capacity constraint
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
            route_load[rx] = route_load[ry] + demands[rx] + demands[ry]
            return True
        
        # Initialize: each customer is its own route
        for i in range(1, n):
            route_load[i] = demands[i]
        
        # Process pairs in order
        for s_mod, i, j in pairs:
            if time.time() > deadline:
                break
            
            ri, rj = find(i), find(j)
            if ri == rj:
                continue
            
            # Check capacity constraint
            new_load = route_load[ri] + route_load[rj]
            if new_load > capacity:
                continue
            
            if union(ri, rj):
                edge_counts[i, j] += 1.0
                edge_counts[j, i] += 1.0
        
        # After each run, also record the edges in the final routes
        # For each route, add edges between consecutive customers
        # This is a simplification - we just use the union edges
    
    # Use log-sum-exp style smooth aggregation of edge counts
    # This provides a smooth surrogate for the max-count objective
    # boost = exp(count / temperature) normalized
    
    temperature = 2.0
    if edge_counts.max() > 0:
        # Normalize counts
        normalized_counts = edge_counts / edge_counts.max()
        # Soft-max via exp
        boost = np.exp(normalized_counts / temperature)
        # Normalize boost so that average boost is around 1
        avg_boost = boost.mean()
        if avg_boost > 0:
            boost = boost / avg_boost
        # Apply boost to heuristic
        heuristic = heuristic * boost
    
    # Final cleanup
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    return heuristic
