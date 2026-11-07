import numpy as np
import time
import heapq
from collections import defaultdict

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes a heuristic matrix for ACO.
    Strategy:
    1. Use a multi-oracle ensemble: run several diverse constructive heuristics
       (nearest neighbor, savings, angle-based, demand-aware) to generate many
       candidate routes.
    2. Apply local search (2-opt + or-opt) to improve each route.
    3. Aggregate edge frequencies across all good solutions, weighted by solution quality.
    4. Construct the heuristic matrix from this aggregated edge attractiveness,
       blended with a base distance heuristic to ensure coverage.
    This directly encodes structural knowledge of good solutions rather than
       relying on a parametric formula tuned via simulation.
    """
    start_time = time.time()
    deadline = start_time + 315.0  # Safety margin before 330s kill

    n = len(demands)
    demands = np.array(demands, dtype=float)
    dist_mat = np.array(distance_matrix, dtype=float)
    coords = np.array(coordinates, dtype=float)

    # Precompute angular coordinates for angle-based heuristics
    depot_coord = coords[0]
    angles = np.zeros(n)
    for i in range(1, n):
        dx = coords[i, 0] - depot_coord[0]
        dy = coords[i, 1] - depot_coord[1]
        angles[i] = np.arctan2(dy, dx)

    # Helper: compute total route distance for a sequence of nodes (excluding depot returns)
    def route_distance(route):
        """Route is a list of customer indices. Returns total distance including depot visits."""
        if not route:
            return 0.0
        total = dist_mat[0, route[0]]
        for i in range(len(route) - 1):
            total += dist_mat[route[i], route[i + 1]]
        total += dist_mat[route[-1], 0]
        return total

    def construct_nearest_neighbor(seed_offset=0):
        """Nearest neighbor construction, possibly with slight randomization."""
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        current = 0
        load = 0.0
        route = []
        
        while not np.all(visited):
            feasible = [j for j in range(1, n) if not visited[j] and load + demands[j] <= capacity]
            if not feasible:
                current = 0
                load = 0.0
                continue
            
            if seed_offset > 0 and len(feasible) > 1:
                # Pick from top 3 nearest with some randomness
                dists = [dist_mat[current, j] for j in feasible]
                sorted_idx = np.argsort(dists)
                top_k = min(3, len(feasible))
                top_feasible = [feasible[i] for i in sorted_idx[:top_k]]
                # Weighted selection
                weights = [1.0 / (dist_mat[current, top_feasible[k]] + 1e-9) for k in range(top_k)]
                total_w = sum(weights)
                weights = [w / total_w for w in weights]
                r = np.random.random()
                cumsum = 0
                chosen = top_feasible[-1]
                for k, w in enumerate(weights):
                    cumsum += w
                    if r <= cumsum:
                        chosen = top_feasible[k]
                        break
            else:
                chosen = min(feasible, key=lambda j: dist_mat[current, j])
            
            route.append(chosen)
            load += demands[chosen]
            visited[chosen] = True
            current = chosen
        
        if current != 0:
            # Return to depot
            pass
        return route

    def construct_savings():
        """Clarke-Wright savings heuristic."""
        # Start with each customer as its own route
        routes = []
        for i in range(1, n):
            if demands[i] <= capacity:
                routes.append([i])
        
        # Compute savings
        savings = []
        for i in range(1, n):
            for j in range(i + 1, n):
                s = dist_mat[0, i] + dist_mat[0, j] - dist_mat[i, j]
                savings.append((s, i, j))
        
        savings.sort(reverse=True)
        
        # Track which route each node is in
        node_route = {i: i for i in range(1, n)}
        route_members = {i: [i] for i in range(1, n)}
        
        for s, i, j in savings:
            if s <= 0:
                break
            ri = node_route.get(i)
            rj = node_route.get(j)
            if ri is None or rj is None or ri == rj:
                continue
            
            # Check capacity
            ri_members = route_members[ri]
            rj_members = route_members[rj]
            ri_demand = sum(demands[x] for x in ri_members)
            rj_demand = sum(demands[x] for x in rj_members)
            if ri_demand + rj_demand > capacity:
                continue
            
            # Check if i and j are at ends of their routes
            if not (ri_members[0] == i or ri_members[-1] == i):
                continue
            if not (rj_members[0] == j or rj_members[-1] == j):
                continue
            
            # Merge
            # Determine orientation
            if ri_members[-1] == i:
                ri_members = ri_members  # i is at end
            else:
                ri_members = ri_members[::-1]  # i is at start, reverse
            
            if rj_members[0] == j:
                rj_members = rj_members  # j is at start
            else:
                rj_members = rj_members[::-1]  # j is at end, reverse
            
            new_members = ri_members + rj_members
            new_route_id = ri
            node_route[new_route_id] = new_route_id
            for x in rj_members:
                node_route[x] = new_route_id
            route_members[new_route_id] = new_members
            del route_members[rj]
        
        # Flatten routes
        all_customers = list(range(1, n))
        final_routes = []
        for rid in route_members:
            members = route_members[rid]
            if members:
                final_routes.append(members)
        
        # Ensure all customers are covered
        covered = set()
        for r in final_routes:
            covered.update(r)
        for c in all_customers:
            if c not in covered:
                final_routes.append([c])
        
        return final_routes

    def two_opt_route(route):
        """Improve a single route using 2-opt."""
        if len(route) < 2:
            return route
        
        improved = True
        best_cost = route_distance(route)
        current_route = route[:]
        
        while improved:
            improved = False
            for i in range(len(current_route) - 1):
                for j in range(i + 1, len(current_route)):
                    # Try reversing segment i..j
                    new_route = current_route[:i] + current_route[i:j+1][::-1] + current_route[j+1:]
                    new_cost = route_distance(new_route)
                    if new_cost < best_cost - 1e-9:
                        best_cost = new_cost
                        current_route = new_route
                        improved = True
        
        return current_route

    def or_opt_route(route):
        """Improve a single route using or-opt (move 1-3 consecutive nodes)."""
        if len(route) < 3:
            return route
        
        improved = True
        best_cost = route_distance(route)
        current_route = route[:]
        
        while improved:
            improved = False
            for L in [1, 2, 3]:
                for i in range(len(current_route) - L):
                    segment = current_route[i:i+L]
                    # Remove segment
                    remaining = current_route[:i] + current_route[i+L:]
                    if not remaining:
                        continue
                    # Try inserting at each position
                    for j in range(len(remaining) + 1):
                        if j == i or j == i + L:
                            continue
                        new_route = remaining[:j] + segment + remaining[j:]
                        new_cost = route_distance(new_route)
                        if new_cost < best_cost - 1e-9:
                            best_cost = new_cost
                            current_route = new_route
                            improved = True
        
        return current_route

    # Generate solutions from multiple oracles
    solutions = []
    
    # Oracle 1: Nearest neighbor
    if time.time() < deadline - 5.0:
        route = construct_nearest_neighbor(0)
        # Split into feasible sub-routes (already handled in construction)
        # Actually, the construction returns a single sequence. We need to split by capacity.
        # Let's redo: the construction already handles capacity by returning to depot.
        # But the route list is just the sequence of customers. We need to identify route breaks.
        # Let's modify: track route segments during construction.
        
    def construct_nearest_neighbor_segments(seed_offset=0):
        """Returns list of route segments (each a list of customers)."""
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        current = 0
        load = 0.0
        segments = []
        current_segment = []
        
        while not np.all(visited):
            feasible = [j for j in range(1, n) if not visited[j] and load + demands[j] <= capacity]
            if not feasible:
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
                current = 0
                load = 0.0
                continue
            
            if seed_offset > 0 and len(feasible) > 1:
                dists = [dist_mat[current, j] for j in feasible]
                sorted_idx = np.argsort(dists)
                top_k = min(3, len(feasible))
                top_feasible = [feasible[i] for i in sorted_idx[:top_k]]
                weights = [1.0 / (dist_mat[current, top_feasible[k]] + 1e-9) for k in range(top_k)]
                total_w = sum(weights)
                weights = [w / total_w for w in weights]
                r = np.random.random()
                cumsum = 0
                chosen = top_feasible[-1]
                for k, w in enumerate(weights):
                    cumsum += w
                    if r <= cumsum:
                        chosen = top_feasible[k]
                        break
            else:
                chosen = min(feasible, key=lambda j: dist_mat[current, j])
            
            current_segment.append(chosen)
            load += demands[chosen]
            visited[chosen] = True
            current = chosen
        
        if current_segment:
            segments.append(current_segment)
        
        return segments

    # Generate multiple solutions
    np.random.seed(42)
    
    # Nearest neighbor variants
    for seed_offset in [0, 1, 2, 3, 4]:
        if time.time() > deadline - 5.0:
            break
        segments = construct_nearest_neighbor_segments(seed_offset)
        # Improve each segment
        improved_segments = []
        for seg in segments:
            seg = two_opt_route(seg)
            seg = or_opt_route(seg)
            improved_segments.append(seg)
        solutions.append(improved_segments)
    
    # Savings heuristic
    if time.time() < deadline - 5.0:
        segments = construct_savings()
        improved_segments = []
        for seg in segments:
            seg = two_opt_route(seg)
            seg = or_opt_route(seg)
            improved_segments.append(seg)
        solutions.append(improved_segments)
    
    # Angle-based construction
    if time.time() < deadline - 5.0:
        # Sort customers by angle, then greedily assign to routes
        customers = list(range(1, n))
        customers.sort(key=lambda x: angles[x])
        
        routes = []
        route_demands = []
        
        for c in customers:
            placed = False
            for r_idx in range(len(routes)):
                if route_demands[r_idx] + demands[c] <= capacity:
                    routes[r_idx].append(c)
                    route_demands[r_idx] += demands[c]
                    placed = True
                    break
            if not placed:
                routes.append([c])
                route_demands.append(demands[c])
        
        improved_segments = []
        for seg in routes:
            seg = two_opt_route(seg)
            seg = or_opt_route(seg)
            improved_segments.append(seg)
        solutions.append(improved_segments)
    
    # Demand-aware: prioritize high-demand nodes early
    if time.time() < deadline - 5.0:
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        current = 0
        load = 0.0
        segments = []
        current_segment = []
        
        while not np.all(visited):
            feasible = [j for j in range(1, n) if not visited[j] and load + demands[j] <= capacity]
            if not feasible:
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
                current = 0
                load = 0.0
                continue
            
            # Score: balance distance and demand (prefer high demand if close)
            scores = []
            for j in feasible:
                d = dist_mat[current, j]
                dem = demands[j]
                score = d / (1.0 + 0.1 * dem)
                scores.append((score, j))
            scores.sort()
            chosen = scores[0][1]
            
            current_segment.append(chosen)
            load += demands[chosen]
            visited[chosen] = True
            current = chosen
        
        if current_segment:
            segments.append(current_segment)
        
        improved_segments = []
        for seg in segments:
            seg = two_opt_route(seg)
            seg = or_opt_route(seg)
            improved_segments.append(seg)
        solutions.append(improved_segments)
    
    # Aggregate edge frequencies
    edge_weight = np.zeros((n, n))
    
    for sol in solutions:
        # Compute total cost of this solution
        total_cost = 0.0
        for seg in sol:
            total_cost += route_distance(seg)
        
        if total_cost <= 0:
            continue
        
        # Weight edges by inverse cost (better solutions contribute more)
        weight = 1.0 / total_cost
        
        for seg in sol:
            # Add edges within the segment
            for i in range(len(seg) - 1):
                u = seg[i]
                v = seg[i + 1]
                edge_weight[u, v] += weight
                edge_weight[v, u] += weight
            # Add depot edges
            if seg:
                edge_weight[0, seg[0]] += weight
                edge_weight[seg[0], 0] += weight
                edge_weight[seg[-1], 0] += weight
                edge_weight[0, seg[-1]] += weight
    
    # Build heuristic matrix: blend edge frequency with base distance heuristic
    # Base: 1 / (distance + epsilon)
    safe_dist = dist_mat + 1e-9
    base_eta = 1.0 / safe_dist
    
    # Normalize edge_weight to comparable scale
    max_edge_weight = np.max(edge_weight)
    if max_edge_weight > 0:
        normalized_edge = edge_weight / max_edge_weight
    else:
        normalized_edge = np.zeros_like(edge_weight)
    
    # Blend: eta = alpha * base_eta + (1 - alpha) * normalized_edge
    # Use alpha = 0.3 to give more weight to learned structure
    alpha = 0.3
    eta = alpha * base_eta + (1.0 - alpha) * normalized_edge
    
    # Ensure all entries are positive and finite
    eta[~np.isfinite(eta)] = 1e-9
    eta[eta < 1e-9] = 1e-9
    
    # If time permits, run a quick ACO simulation to verify and potentially adjust
    if time.time() < deadline - 10.0:
        def simulate_aco(eta_matrix, seed=42):
            rng = np.random.default_rng(seed)
            tau = np.ones((n, n))
            best_route_cost = float('inf')
            
            for it in range(30):
                iteration_best_cost = float('inf')
                
                for ant in range(20):
                    current = 0
                    load = 0.0
                    visited = np.zeros(n, dtype=bool)
                    visited[0] = True
                    total_dist = 0.0
                    
                    while not np.all(visited):
                        feasible = [j for j in range(1, n) if not visited[j] and load + demands[j] <= capacity]
                        
                        if not feasible:
                            total_dist += dist_mat[current, 0]
                            current = 0
                            load = 0.0
                            continue
                        
                        probs = tau[current, feasible] * eta_matrix[current, feasible]
                        sum_probs = np.sum(probs)
                        if sum_probs == 0:
                            idx = rng.integers(0, len(feasible))
                        else:
                            probs = probs / sum_probs
                            idx = rng.choice(len(feasible), p=probs)
                        
                        next_node = feasible[idx]
                        total_dist += dist_mat[current, next_node]
                        load += demands[next_node]
                        visited[next_node] = True
                        current = next_node
                    
                    if current != 0:
                        total_dist += dist_mat[current, 0]
                    
                    if total_dist < iteration_best_cost:
                        iteration_best_cost = total_dist
                    if total_dist < best_route_cost:
                        best_route_cost = total_dist
                
                tau = 0.9 * tau + 1.0 / iteration_best_cost
            
            return best_route_cost
        
        # Test current eta
        test_cost = simulate_aco(eta, seed=42)
        
        # Try a few alternative blend ratios
        best_eta = eta
        best_cost = test_cost
        
        for alpha_try in [0.1, 0.2, 0.4, 0.5]:
            if time.time() > deadline - 5.0:
                break
            eta_try = alpha_try * base_eta + (1.0 - alpha_try) * normalized_edge
            eta_try[~np.isfinite(eta_try)] = 1e-9
            eta_try[eta_try < 1e-9] = 1e-9
            
            cost_try = simulate_aco(eta_try, seed=42)
            if cost_try < best_cost:
                best_cost = cost_try
                best_eta = eta_try
        
        eta = best_eta
    
    # Final safety check
    if not np.all(np.isfinite(eta)):
        eta = 1.0 / safe_dist
        eta[~np.isfinite(eta)] = 1e-9
        eta[eta < 1e-9] = 1e-9
    
    return eta
