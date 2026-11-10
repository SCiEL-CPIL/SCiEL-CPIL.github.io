import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix (larger = more attractive) for ACO CVRP.
    
    Strategy:
    1. Base heuristic: Power-law inverse distance (1/d^1.5).
    2. Structural boost: Clarke-Wright Savings Heuristic to identify high-value edges.
    3. Depot-Pressure modulation: Boost edges leading to nodes with high 'demand-per-distance'
       from the depot, encouraging early service of high-demand, close-to-depot nodes.
    
    Returns an n x n matrix of the same shape as distance_matrix.
    """
    start_time = time.time()
    deadline = start_time + 280.0

    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    demands = np.asarray(demands, dtype=np.float64)
    
    n = distance_matrix.shape[0]
    if n == 0:
        return np.array([])
    if distance_matrix.shape[1] != n:
        return np.full((n, n), 1e-9)
        
    # Ensure demands matches n
    if demands.shape[0] != n:
        new_demands = np.zeros(n, dtype=np.float64)
        min_len = min(demands.shape[0], n)
        new_demands[:min_len] = demands[:min_len]
        demands = new_demands

    # Safe distance matrix
    dist_safe = np.maximum(distance_matrix, 1e-9)
    
    # 1. Base Heuristic: Power law 1/d^1.5
    base_heur = 1.0 / (dist_safe ** 1.5)
    
    # 2. Depot Pressure Calculation
    # P_j = demand_j / dist(0, j)
    # High P means high demand close to depot. These should be prioritized.
    d0 = dist_safe[0, :]
    # Avoid division by zero for depot itself (though depot demand is 0)
    d0_safe = np.maximum(d0, 1e-9)
    pressure = demands / d0_safe
    
    # Normalize pressure to [0, 1] for scaling
    max_pressure = np.max(pressure)
    if max_pressure > 0:
        pressure_norm = pressure / max_pressure
    else:
        pressure_norm = np.zeros_like(pressure)
    
    # Apply a subtle pressure boost to the base heuristic.
    # Boost edges (i, j) based on the pressure of node j.
    # This encourages visiting high-pressure nodes.
    # Factor: 1 + 0.5 * pressure_norm[j]
    pressure_boost = 1.0 + 0.5 * pressure_norm[None, :]
    base_heur = base_heur * pressure_boost
    
    # 3. Clarke-Wright Savings Heuristic for Structural Boost
    # Calculate savings for all pairs (i, j) with i, j > 0
    # Savings = d(0,i) + d(0,j) - d(i,j)
    
    d0_customers = d0[1:]
    # Matrix of savings: S[i, j] = d0[i] + d0[j] - d[i, j]
    # Indices 1..n-1
    num_customers = n - 1
    if num_customers > 1:
        savings_matrix = d0_customers[:, None] + d0_customers[None, :] - dist_safe[1:, 1:]
        
        # Capacity constraint: demands[i] + demands[j] <= capacity
        dem_customers = demands[1:]
        cap_constraint = (dem_customers[:, None] + dem_customers[None, :]) <= capacity
        
        # Mask invalid pairs
        savings_matrix[~cap_constraint] = -np.inf
        np.fill_diagonal(savings_matrix, -np.inf)
        
        # Sort savings in descending order
        flat_savings = savings_matrix.flatten()
        flat_indices = np.argsort(flat_savings)[::-1]
        
        # Union-Find for route tracking
        parent = np.arange(n)
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
                return True
            return False

        route_demand = demands.copy()
        ref_edges = set()
        merges_done = 0
        
        # Iterate through sorted savings
        for flat_idx in flat_indices:
            if time.time() > deadline:
                break
            if merges_done >= num_customers - 1:
                break
                
            s_val = flat_savings[flat_idx]
            if s_val <= 0:
                break
                
            i_cust = flat_idx // num_customers
            j_cust = flat_idx % num_customers
            
            # Actual node indices
            i = i_cust + 1
            j = j_cust + 1
            
            pi, pj = find(i), find(j)
            if pi == pj:
                continue
            
            # Check capacity
            if route_demand[pi] + route_demand[pj] <= capacity:
                union(i, j)
                new_root = find(i)
                route_demand[new_root] = route_demand[pi] + route_demand[pj]
                
                ref_edges.add((i, j))
                ref_edges.add((j, i))
                merges_done += 1
    else:
        ref_edges = set()

    # 4. Apply Structural Boost
    # Boost edges identified by savings heuristic
    if ref_edges:
        boost_factor = 5.0
        for i, j in ref_edges:
            base_heur[i, j] *= boost_factor
            base_heur[j, i] *= boost_factor

    # 5. Final Sanitization
    # Ensure all entries are strictly positive and finite
    base_heur[~np.isfinite(base_heur)] = 1e-9
    base_heur[base_heur <= 0] = 1e-9

    return base_heur
