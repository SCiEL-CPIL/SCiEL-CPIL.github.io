# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix (larger = more attractive) for ACO CVRP.
    
    Strategy:
    1. Base heuristic: Power-law inverse distance (1/d^1.5) to strongly prefer local moves.
    2. Structural boost: Use a capacity-aware Clarke-Wright Savings Heuristic 
       to identify high-value edges that form good route segments.
    3. Boost these specific edges in the heuristic matrix to guide the ACO.
    
    Returns an n x n matrix of the same shape as distance_matrix.
    """
    start_time = time.time()
    deadline = start_time + 280.0

    n = distance_matrix.shape[0]
    if n == 0:
        return np.array([])
    if distance_matrix.shape[1] != n:
        return np.full((n, n), 1e-9)

    demands = np.asarray(demands, dtype=np.float64)
    if demands.shape[0] != n:
        # Pad or trim if necessary, though contract says fixed 50 customers + depot
        new_demands = np.zeros(n, dtype=np.float64)
        min_len = min(demands.shape[0], n)
        new_demands[:min_len] = demands[:min_len]
        demands = new_demands

    # Safe distance matrix to avoid division by zero
    dist_safe = np.maximum(distance_matrix, 1e-9)
    
    # 1. Base Heuristic: Power law 1/d^1.5
    # This emphasizes short distances more than 1/d, helping ACO find tight local loops
    base_heur = 1.0 / (dist_safe ** 1.5)

    # 2. Savings Heuristic for Edge Boosting
    # We identify edges that form a good solution using Clarke-Wright Savings.
    # These edges are "attractive" because they reduce total distance significantly.
    
    # Initialize routes: each customer is its own route
    # route_rep[i] = the representative (root) of the route containing i
    # For simplicity in tracking edges, we just collect the pairs (i,j) that are merged.
    
    # Calculate savings for all pairs (i, j) with i, j > 0
    # Savings = d(0,i) + d(0,j) - d(i,j)
    # Only valid if demands[i] + demands[j] <= capacity
    
    # Optimization: Vectorize savings calculation
    # d0_i = distance from depot to i
    d0 = dist_safe[0, :]
    
    # Matrix of savings: S[i, j] = d0[i] + d0[j] - d[i, j]
    # We only care about i, j >= 1
    savings_matrix = d0[1:, None] + d0[None, 1:] - dist_safe[1:, 1:]
    
    # Capacity constraint: demands[i] + demands[j] <= capacity
    dem_customers = demands[1:]
    cap_constraint = (dem_customers[:, None] + dem_customers[None, :]) <= capacity
    
    # Mask invalid pairs (where capacity is exceeded or i==j)
    savings_matrix[~cap_constraint] = -np.inf
    np.fill_diagonal(savings_matrix, -np.inf)
    
    # Sort savings in descending order
    # Flatten to get indices
    flat_savings = savings_matrix.flatten()
    flat_indices = np.argsort(flat_savings)[::-1]
    
    # Track which nodes are already merged into a route
    # We use a union-find like structure or simply track the "current route" for each node.
    # For efficiency and simplicity in this heuristic context, we'll track a 'parent' pointer.
    # However, a simpler approach for just getting the *edges* is to maintain a set of active nodes
    # and a dictionary of current route members.
    
    # Let's use a simple Union-Find to manage routes.
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

    # We also need to track the total demand of each route to check capacity on merge
    route_demand = demands.copy()
    
    ref_edges = set()
    
    # Iterate through sorted savings
    # Limit iterations to avoid timeout if n is large, but n=50 is small.
    num_customers = n - 1
    # Max possible merges is num_customers - 1
    merges_done = 0
    
    # Map flat index back to (i, j) in customer space (1..n-1)
    # flat_idx = i_customers * num_customers + j_customers
    # where i_customers, j_customers are 0-indexed in the customer subset
    # Actual node indices: i = i_customers + 1, j = j_customers + 1
    
    for flat_idx in flat_indices:
        if time.time() > deadline:
            break
        if merges_done >= num_customers - 1:
            break
            
        s_val = flat_savings[flat_idx]
        if s_val <= 0: # No more positive savings
            break
            
        i_cust = flat_idx // num_customers
        j_cust = flat_idx % num_customers
        
        # Convert to actual node indices
        i = i_cust + 1
        j = j_cust + 1
        
        # Check if they are in different routes
        pi, pj = find(i), find(j)
        if pi == pj:
            continue
            
        # Check capacity
        if route_demand[pi] + route_demand[pj] <= capacity:
            # Merge
            union(i, j)
            # Update demand of the new root
            new_root = find(i)
            route_demand[new_root] = route_demand[pi] + route_demand[pj]
            
            # Record the edge
            ref_edges.add((i, j))
            ref_edges.add((j, i))
            merges_done += 1

    # 3. Apply Boost
    # Boost the heuristic values for the edges identified by the savings heuristic.
    # A factor of 5.0 is a reasonable balance between guidance and exploration.
    boost_factor = 5.0
    
    if ref_edges:
        # Vectorized update
        # Create a mask or just loop if few edges. With n=50, edges are ~50.
        for i, j in ref_edges:
            base_heur[i, j] *= boost_factor
            base_heur[j, i] *= boost_factor

    # 4. Final Sanitization
    # Ensure all entries are strictly positive and finite
    base_heur[~np.isfinite(base_heur)] = 1e-9
    base_heur[base_heur <= 0] = 1e-9

    return base_heur
# EVOLVE-BLOCK-END
