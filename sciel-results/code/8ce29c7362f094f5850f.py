import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    start_time = time.time()
    deadline = start_time + 250.0  # Safety margin
    
    n = distance_matrix.shape[0]
    customers = np.arange(1, n)
    num_customers = n - 1
    
    # 1. Base Heuristic: Inverse Distance
    eps = 1e-9
    base_heur = 1.0 / (distance_matrix + eps)
    np.fill_diagonal(base_heur, 0.0)
    
    # 2. Savings Heuristic Signal
    d_to_depot = distance_matrix[:, 0]
    savings = d_to_depot[:, None] + d_to_depot[None, :] - distance_matrix
    cust_savings = savings[np.ix_(customers, customers)]
    cust_savings = np.clip(cust_savings, 0, None)
    
    max_savings = np.max(cust_savings) if num_customers > 0 else 1.0
    if max_savings > 0:
        norm_savings = cust_savings / max_savings
    else:
        norm_savings = np.zeros_like(cust_savings)
    
    # 3. Lagrangian Relaxation Heuristic
    # We estimate edge attractiveness by solving a relaxed TSP-like problem
    # where capacity constraints are penalized via dual variables.
    
    # Estimate number of routes
    total_demand = np.sum(demands[1:])
    num_routes = max(1, int(np.ceil(total_demand / capacity)))
    num_routes = min(num_routes, num_customers)
    
    # Initialize Lagrangian multipliers (dual variables for capacity)
    # We use a simple subgradient method to update these.
    
    # For each route, we maintain a "load" variable.
    # The Lagrangian objective is: sum(d_ij * x_ij) + sum_u lambda_u * (load_u - capacity)
    
    # Since we don't have a full MIP solver, we approximate the relaxed solution
    # by finding a set of routes that minimize distance while penalizing load violations.
    
    # A simpler approach: Use a "virtual" cost that includes a penalty for 
    # being in a route that is "too full". But this requires knowing the routes.
    
    # Alternative: Use the structure of the problem.
    # Edges that are part of many good savings-based tours are attractive.
    # Let's run a multi-start savings algorithm and count edge frequencies.
    
    edge_frequency = np.zeros((n, n))
    
    # Number of restarts for savings heuristic
    num_restarts = min(50, max(10, num_customers // 2))
    
    # Time budget for this part
    time_limit_restarts = 10.0
    restart_deadline = start_time + time_limit_restarts
    
    for r in range(num_restarts):
        if time.time() > restart_deadline:
            break
            
        # Randomize savings order slightly
        # Create list of (savings, i, j) for i < j
        savings_list = []
        for i in range(num_customers):
            ci = customers[i]
            for j in range(i + 1, num_customers):
                cj = customers[j]
                s = cust_savings[i, j]
                if s > 0:
                    # Add some random noise to break ties and diversify
                    noise = np.random.uniform(0, 0.1 * s) if s > 0 else 0
                    savings_list.append((s + noise, i, j))
        
        # Sort by savings descending
        savings_list.sort(key=lambda x: -x[0])
        
        # Initialize routes: each customer is its own route
        # route[customer_idx] = list of customers in route
        # We use indices 0..num_customers-1 for customers, mapping to customers[idx]
        
        # parent[i] = parent of i in the route tree (or -1 if root)
        # We use a disjoint set union (DSU) to manage routes
        parent = list(range(num_customers))
        load = [demands[customers[i]] for i in range(num_customers)]
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            rx = find(x)
            ry = find(y)
            if rx != ry:
                if load[rx] + load[ry] <= capacity:
                    parent[ry] = rx
                    load[rx] += load[ry]
        
        # Apply savings
        for s, i, j in savings_list:
            if time.time() > restart_deadline:
                break
            ri = find(i)
            rj = find(j)
            if ri != rj:
                if load[ri] + load[rj] <= capacity:
                    union(i, j)
        
        # Count edges in the resulting routes
        # For each root, the route is a path. We need to reconstruct the path.
        # Actually, the savings algorithm merges routes but doesn't explicitly store the path order.
        # However, we can infer that if i and j were merged, the edge (ci, cj) is in the tour.
        # This is an approximation. A more rigorous way is to store the edges.
        
        # Let's redo this by storing edges explicitly.
        
    # Redo with explicit edge storage
    edge_frequency = np.zeros((n, n))
    
    for r in range(num_restarts):
        if time.time() > restart_deadline:
            break
        
        savings_list = []
        for i in range(num_customers):
            ci = customers[i]
            for j in range(i + 1, num_customers):
                cj = customers[j]
                s = cust_savings[i, j]
                if s > 0:
                    noise = np.random.uniform(0, 0.1 * s) if s > 0 else 0
                    savings_list.append((s + noise, i, j))
        
        savings_list.sort(key=lambda x: -x[0])
        
        parent = list(range(num_customers))
        load = [demands[customers[i]] for i in range(num_customers)]
        edges_in_tour = set()
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            rx = find(x)
            ry = find(y)
            if rx != ry:
                if load[rx] + load[ry] <= capacity:
                    parent[ry] = rx
                    load[rx] += load[ry]
                    # Add edge between the roots' representatives? No, add edge between x and y?
                    # In savings, we connect the two routes. The specific edge is (ci, cj).
                    edges_in_tour.add((customers[x], customers[y]))
                    edges_in_tour.add((customers[y], customers[x]))
        
        for s, i, j in savings_list:
            if time.time() > restart_deadline:
                break
            ri = find(i)
            rj = find(j)
            if ri != rj:
                if load[ri] + load[rj] <= capacity:
                    union(i, j)
        
        # Update frequency
        for u, v in edges_in_tour:
            edge_frequency[u, v] += 1
    
    # Normalize edge frequency
    max_freq = np.max(edge_frequency) if num_customers > 0 else 1.0
    if max_freq > 0:
        norm_freq = edge_frequency / max_freq
    else:
        norm_freq = np.zeros_like(edge_frequency)
    
    # 4. Combine Heuristics
    # H = Base * (1 + alpha * Savings + beta * Freq)
    alpha = 1.0
    beta = 1.0
    
    final_heur = base_heur.copy()
    
    # Apply boosts to customer-customer edges
    boost = 1.0 + alpha * norm_savings + beta * norm_freq[np.ix_(customers, customers)]
    
    full_boost = np.ones((n, n))
    full_boost[np.ix_(customers, customers)] = boost
    
    final_heur = final_heur * full_boost
    
    # Ensure positivity and finiteness
    final_heur = np.where(final_heur > 0, final_heur, 1e-9)
    final_heur = np.where(np.isfinite(final_heur), final_heur, 1e-9)
    
    if final_heur.shape != distance_matrix.shape:
        final_heur = 1.0 / (distance_matrix + eps)
        
    return final_heur
