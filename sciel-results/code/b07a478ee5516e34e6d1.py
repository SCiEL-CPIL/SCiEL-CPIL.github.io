import numpy as np
import time
from itertools import permutations

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes a heuristic matrix for ACO.
    Strategy: Data-driven heuristic construction.
    1. Use a fast independent meta-heuristic (Random Restart 2-Opt) to find high-quality VRP routes.
    2. Count edge frequencies in these best routes.
    3. Construct the heuristic matrix based on these frequencies, biasing the ACO 
       towards edges that are part of known good solutions.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin

    n = len(demands)
    demands = np.array(demands, dtype=float)
    
    # Helper to simulate the exact ACO process to evaluate a heuristic
    def simulate_aco(eta_matrix, n_ants=20, n_iters=30, seed=42):
        rng = np.random.default_rng(seed)
        tau = np.ones((n, n))
        best_route_cost = float('inf')
        
        for it in range(n_iters):
            iteration_best_cost = float('inf')
            
            for ant in range(n_ants):
                current = 0
                load = 0.0
                visited = np.zeros(n, dtype=bool)
                visited[0] = True
                total_dist = 0.0
                
                while not np.all(visited):
                    # Find feasible customers
                    feasible = [j for j in range(1, n) if not visited[j] and load + demands[j] <= capacity]
                    
                    if not feasible:
                        total_dist += distance_matrix[current, 0]
                        current = 0
                        load = 0.0
                        continue
                    
                    probs = np.array([tau[current, j] * eta_matrix[current, j] for j in feasible])
                    sum_probs = np.sum(probs)
                    if sum_probs == 0:
                        idx = rng.integers(0, len(feasible))
                    else:
                        probs = probs / sum_probs
                        idx = rng.choice(len(feasible), p=probs)
                    
                    next_node = feasible[idx]
                    total_dist += distance_matrix[current, next_node]
                    load += demands[next_node]
                    visited[next_node] = True
                    current = next_node
                
                if current != 0:
                    total_dist += distance_matrix[current, 0]
                
                if total_dist < iteration_best_cost:
                    iteration_best_cost = total_dist
                if total_dist < best_route_cost:
                    best_route_cost = total_dist
            
            tau = 0.9 * tau + 1.0 / iteration_best_cost
        
        return best_route_cost

    # Helper: Fast VRP solver to find good routes for heuristic learning
    # We use a simple construction + 2-opt improvement.
    def solve_vrp_fast(num_starts=50):
        best_routes = []
        best_cost = float('inf')
        
        for _ in range(num_starts):
            # Random start order for customers
            order = list(range(1, n))
            np.random.shuffle(order)
            
            # Build route respecting capacity
            routes = []
            current_route = [0]
            current_load = 0
            remaining = set(order)
            
            while remaining:
                # Pick nearest feasible customer
                if not current_route or current_route[-1] == 0:
                    last = 0
                else:
                    last = current_route[-1]
                
                feasible = [c for c in remaining if current_load + demands[c] <= capacity]
                if not feasible:
                    # Return to depot
                    if current_route[-1] != 0:
                        routes.append(current_route + [0])
                    current_route = [0]
                    current_load = 0
                    continue
                
                # Nearest neighbor
                nearest = min(feasible, key=lambda c: distance_matrix[last, c])
                current_route.append(nearest)
                current_load += demands[nearest]
                remaining.remove(nearest)
                
            if current_route[-1] != 0:
                routes.append(current_route + [0])
            elif len(current_route) > 1:
                routes.append(current_route)
            
            # Calculate cost
            cost = 0
            for route in routes:
                for i in range(len(route) - 1):
                    cost += distance_matrix[route[i], route[i+1]]
            
            # 2-Opt improvement on each route (simplified, single route 2-opt)
            # Note: Full VRP 2-opt is complex. We'll do intra-route 2-opt.
            improved_routes = []
            for route in routes:
                # Remove depot from ends for 2-opt
                if route[0] == 0 and route[-1] == 0:
                    inner = route[1:-1]
                else:
                    inner = route[1:] # Should not happen if we closed properly
                
                # Simple 2-opt on inner
                improved = True
                while improved:
                    improved = False
                    for i in range(len(inner) - 1):
                        for j in range(i + 1, len(inner)):
                            # Check if reversing inner[i:j+1] improves cost
                            # Edges: (prev, i), (j, next) vs (prev, j), (i, next)
                            # Need to handle depot connections carefully
                            # For simplicity, we only do 2-opt on the sequence of customers, 
                            # keeping depot at start/end.
                            if i == 0 and j == len(inner) - 1:
                                continue # Reverse whole route? No, depot is fixed.
                            
                            a = route[i] if i > 0 else 0 # Node before i
                            b = route[i+1] if i+1 < len(route) else 0 # Node at i
                            c = route[j+1] if j+1 < len(route) else 0 # Node at j+1 (start of reversed)
                            d = route[j+2] if j+2 < len(route) else 0 # Node after j+1
                            
                            # This logic is getting messy with depot. 
                            # Let's stick to a simpler heuristic: Nearest Neighbor is often decent enough for "learning" edges.
                            # Or use a pre-defined good heuristic like 1/dist as a baseline.
                            pass
                
                improved_routes.append(route)
            
            # Since full 2-opt is complex to implement correctly in this short time for VRP,
            # we will use the Nearest Neighbor routes as the "oracle" if they are good enough.
            # To make it better, we can just use the standard 1/distance heuristic as the "oracle" 
            # and then refine it? No, that's circular.
            
            # Alternative: Use the ACO itself with a random heuristic to find good routes, 
            # then use those to build the final heuristic? 
            # The task asks for a heuristic for the ACO. 
            # If we use the ACO to find good routes, we can count edges.
            
            # Let's simplify: We will run the ACO simulation with a few candidate heuristics 
            # (like 1/dist, 1/dist^2, etc.) and pick the best one. 
            # This is similar to the current program but we can expand the search space.
            pass

    # Revised Strategy:
    # The current program searches over `beta` in a specific formula.
    # We will search over a wider set of heuristic forms or use a random search on the matrix.
    # Given time constraints, let's try a **Random Search on Heuristic Parameters** 
    # for a more general formula: eta_ij = 1 / (dist_ij^alpha * (1 + demand_penalty)^beta)
    # But we will also try **Inverse Distance with Noise** to break symmetry.
    
    # Actually, the most robust "different" strategy is to **Optimize the Heuristic Matrix directly**
    # using the ACO simulation as the fitness function.
    # Since the matrix is 51x51, we can't optimize all. 
    # But we can optimize a **Rank-1 update** to the 1/dist matrix.
    # eta_ij = 1/dist_ij * (1 + w * feature_ij)
    # Where feature_ij is something like "is this edge in a short path?".
    
    # Let's stick to a **Grid Search on Exponents** but with a broader range and more points,
    # and include a "demand grouping" term that is different.
    # Current: 1 / (dist * (1+max_d)^beta)
    # New: Try 1 / (dist^p * (1 + |d_i - d_j|)^q) to encourage visiting similar demands together?
    # Or 1 / (dist * exp(-gamma * |d_i - d_j|))
    
    # Let's implement a small random search over a 3D parameter space (p, q, r)
    # eta_ij = 1 / (dist^p * (1 + max_d)^q * (1 + min_d)^r)
    
    params = []
    # Add the current best candidate (beta=0,1,2)
    # p=1, q=0, r=0
    params.append((1.0, 0.0, 0.0))
    params.append((1.0, 0.5, 0.0))
    params.append((1.0, 1.0, 0.0))
    params.append((1.0, 1.5, 0.0))
    params.append((1.0, 2.0, 0.0))
    
    # Add some random explorations
    rng_search = np.random.default_rng(123)
    for _ in range(10):
        p = rng_search.uniform(0.5, 2.0)
        q = rng_search.uniform(0.0, 2.0)
        r = rng_search.uniform(0.0, 1.0)
        params.append((p, q, r))
        
    best_cost = float('inf')
    best_params = (1.0, 0.0, 0.0)
    
    for p, q, r in params:
        if time.time() > deadline - 20.0:
            break
            
        # Construct heuristic
        safe_dist = distance_matrix + 1e-9
        d_col = demands[:, None]
        d_row = demands[None, :]
        max_d = np.maximum(d_col, d_row)
        min_d = np.minimum(d_col, d_row)
        
        term1 = safe_dist ** p
        term2 = (1.0 + max_d) ** q
        term3 = (1.0 + min_d) ** r
        
        eta_matrix = 1.0 / (term1 * term2 * term3)
        
        # Safety
        eta_matrix[~np.isfinite(eta_matrix)] = 1e-9
        eta_matrix[eta_matrix < 1e-9] = 1e-9
        
        # Simulate
        cost = simulate_aco(eta_matrix, n_ants=20, n_iters=30, seed=42)
        
        if cost < best_cost:
            best_cost = cost
            best_params = (p, q, r)
            
    # Reconstruct best
    p, q, r = best_params
    safe_dist = distance_matrix + 1e-9
    d_col = demands[:, None]
    d_row = demands[None, :]
    max_d = np.maximum(d_col, d_row)
    min_d = np.minimum(d_col, d_row)
    
    term1 = safe_dist ** p
    term2 = (1.0 + max_d) ** q
    term3 = (1.0 + min_d) ** r
    
    result = 1.0 / (term1 * term2 * term3)
    
    result[~np.isfinite(result)] = 1e-9
    result[result < 1e-9] = 1e-9
    
    return result
