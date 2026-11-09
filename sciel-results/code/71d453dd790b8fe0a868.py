import numpy as np
import time
import math
import random

def plan(instance):
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s kill

    coords = np.array(instance["coords"], dtype=float)
    n_nodes = len(coords)
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    stations = set(instance["stations"])
    station_list = list(instance["stations"])
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    # Precompute distance matrix
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
            dist[i][j] = d
            dist[j][i] = d

    def simulate_route(route):
        """
        Simulates a route (list of nodes, excluding depot start/end).
        Returns (total_distance, total_time, is_feasible)
        """
        if not route:
            return 0.0, 0.0, True

        curr_node = 0
        curr_time = 0.0
        curr_bat = float(battery_cap)
        total_dist = 0.0
        total_demand = 0

        for node in route:
            d = dist[curr_node, node]
            curr_time += d / speed
            curr_bat -= d
            total_dist += d

            if curr_bat < -1e-9:
                return total_dist, curr_time, False

            # Arrive at node
            if node in stations:
                # Charge
                charge_amount = battery_cap - curr_bat
                if charge_amount > 1e-9:
                    charge_time = charge_amount * recharge_rate
                    curr_time += charge_time
                    curr_bat = battery_cap
            else:
                # Customer
                if node <= n_customers:
                    total_demand += demands[node]
                    if total_demand > capacity:
                        return total_dist, curr_time, False

                    # Time window
                    if curr_time < ready[node]:
                        curr_time = ready[node]
                    curr_time += service[node]

                    if curr_time > due[node] + 1e-9:
                        return total_dist, curr_time, False

            curr_node = node

        # Return to depot
        d = dist[curr_node, 0]
        curr_time += d / speed
        curr_bat -= d
        total_dist += d

        if curr_bat < -1e-9:
            return total_dist, curr_time, False
        if curr_time > horizon + 1e-9:
            return total_dist, curr_time, False

        return total_dist, curr_time, True

    def is_feasible(route):
        _, _, feas = simulate_route(route)
        return feas

    def route_distance(route):
        """Fast distance-only calculation (no feasibility check)."""
        if not route:
            return 0.0
        total = dist[0, route[0]]
        for i in range(len(route) - 1):
            total += dist[route[i], route[i + 1]]
        total += dist[route[-1], 0]
        return total

    # === CONSTRUCTION: Cluster-based approach ===
    # 1. Estimate number of routes based on capacity and total demand
    total_demand = sum(demands[c] for c in customers)
    min_routes = max(1, int(math.ceil(total_demand / capacity)))
    
    # Also consider distance: estimate routes needed based on average distance
    avg_cust_dist = np.mean([dist[0, c] for c in customers])
    # Rough estimate: each route can cover ~2*battery_cap/speed distance round trip
    # But with charging, it's more complex. Use min_routes as starting point.
    # Try a few values and pick the best feasible one.
    
    def construct_kmeans_routes(k, seed=42):
        """Construct k routes using k-means clustering, then optimize each."""
        rng = random.Random(seed)
        
        # Initialize centers: pick k random customers
        centers = rng.sample(customers, min(k, len(customers)))
        if len(centers) < k:
            # Pad with depot
            while len(centers) < k:
                centers.append(0)
        
        for iteration in range(20):
            # Assign customers to nearest center
            assignment = [[] for _ in range(k)]
            for c in customers:
                best_center = 0
                best_dist = float('inf')
                for ci, center in enumerate(centers):
                    d = dist[c, center]
                    if d < best_dist:
                        best_dist = d
                        best_center = ci
                assignment[best_center].append(c)
            
            # Update centers
            new_centers = []
            for ci in range(k):
                if assignment[ci]:
                    # Use mean position
                    cx = np.mean([coords[c][0] for c in assignment[ci]])
                    cy = np.mean([coords[c][1] for c in assignment[ci]])
                    # Find nearest customer to this point
                    best_c = assignment[ci][0]
                    best_d = float('inf')
                    for c in assignment[ci]:
                        d = (coords[c][0] - cx) ** 2 + (coords[c][1] - cy) ** 2
                        if d < best_d:
                            best_d = d
                            best_c = c
                    new_centers.append(best_c)
                else:
                    new_centers.append(centers[ci])
            
            if new_centers == centers:
                break
            centers = new_centers
        
        # Build routes from assignments
        routes = []
        for ci in range(k):
            if assignment[ci]:
                # Sort by angle from depot for initial ordering
                def angle_key(c):
                    return math.atan2(coords[c][1] - coords[0][1], coords[c][0] - coords[0][0])
                assignment[ci].sort(key=angle_key)
                routes.append(assignment[ci])
            else:
                routes.append([])
        
        # Remove empty routes
        routes = [r for r in routes if r]
        
        return routes

    def optimize_route(route, max_iters=500):
        """Apply 2-opt and Or-opt to a single route."""
        if len(route) < 2:
            return route
        
        best_route = list(route)
        best_dist = route_distance(best_route)
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            # 2-opt
            n = len(best_route)
            for i in range(n - 1):
                for j in range(i + 1, n):
                    if n - 1 - j < 2 and j == n - 1:
                        continue
                    # Reverse segment i..j
                    new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                    
                    # Quick distance check without full simulation
                    d_new = route_distance(new_route)
                    if d_new < best_dist - 1e-9:
                        # Check feasibility
                        if is_feasible(new_route):
                            best_route = new_route
                            best_dist = d_new
                            improved = True
            
            # Or-opt: move 1, 2, or 3 consecutive nodes to another position
            n = len(best_route)
            for seg_len in [1, 2, 3]:
                for i in range(n - seg_len + 1):
                    segment = best_route[i:i+seg_len]
                    # Remove segment
                    remaining = best_route[:i] + best_route[i+seg_len:]
                    if not remaining:
                        continue
                    # Try inserting at all positions
                    for j in range(len(remaining) + 1):
                        if j == i and j == 0:
                            continue  # Same position
                        new_route = remaining[:j] + segment + remaining[j:]
                        d_new = route_distance(new_route)
                        if d_new < best_dist - 1e-9:
                            if is_feasible(new_route):
                                best_route = new_route
                                best_dist = d_new
                                improved = True
                                break
        
        return best_route

    # Try different k values and pick the best
    best_routes = None
    best_total_dist = float('inf')
    
    for k in range(min_routes, min_routes + 4):
        if time.time() > deadline - 30:
            break
        routes = construct_kmeans_routes(k)
        
        # Optimize each route
        optimized_routes = []
        for r in routes:
            if time.time() > deadline - 20:
                break
            opt_r = optimize_route(r)
            optimized_routes.append(opt_r)
        
        # Check if all customers are covered
        all_visited = set()
        for r in optimized_routes:
            for c in r:
                if c in all_visited:
                    # Duplicate, shouldn't happen but check
                    continue
                all_visited.add(c)
        
        if len(all_visited) != n_customers:
            # Some customers missing, skip this k
            continue
        
        # Check feasibility of all routes
        total_d = 0
        feasible = True
        for r in optimized_routes:
            d, t, f = simulate_route(r)
            if not f:
                feasible = False
                break
            total_d += d
        
        if feasible and total_d < best_total_dist:
            best_total_dist = total_d
            best_routes = optimized_routes
    
    # Fallback: if no feasible solution found, use simple per-customer routes
    if best_routes is None:
        best_routes = [[c] for c in customers]
    
    # === LOCAL SEARCH: Inter-route moves ===
    current_dist = best_total_dist
    
    def try_relocate(route_from, idx_from, route_to):
        """Try to move a single customer from route_from to route_to."""
        cust = route_from[idx_from]
        new_route_from = route_from[:idx_from] + route_from[idx_from+1:]
        d_from_new = route_distance(new_route_from)
        
        # Try all insertion positions in route_to
        best_new_route_to = None
        best_d_to = float('inf')
        
        for pos in range(len(route_to) + 1):
            candidate = route_to[:pos] + [cust] + route_to[pos:]
            d_to = route_distance(candidate)
            if d_to < best_d_to:
                best_d_to = d_to
                best_new_route_to = candidate
        
        if best_new_route_to is None:
            return None, None, None, None
        
        # Check feasibility
        if not is_feasible(best_new_route_to):
            return None, None, None, None
        if not is_feasible(new_route_from):
            return None, None, None, None
        
        return new_route_from, best_new_route_to, d_from_new + best_d_to, (route_from, route_to)

    def try_swap(route1, idx1, route2, idx2):
        """Try to swap two customers between routes."""
        c1 = route1[idx1]
        c2 = route2[idx2]
        
        new_route1 = route1[:idx1] + [c2] + route1[idx1+1:]
        new_route2 = route2[:idx2] + [c1] + route2[idx2+1:]
        
        if not is_feasible(new_route1) or not is_feasible(new_route2):
            return None, None
        
        return new_route1, new_route2

    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        
        # Relocate moves
        for i in range(len(best_routes)):
            if time.time() > deadline - 10:
                break
            route_i = best_routes[i]
            for pos in range(len(route_i)):
                if time.time() > deadline - 10:
                    break
                cust = route_i[pos]
                for j in range(len(best_routes)):
                    if j == i or time.time() > deadline - 10:
                        continue
                    route_j = best_routes[j]
                    
                    # Try inserting cust into route_j at all positions
                    best_new_j = None
                    best_d_j = float('inf')
                    new_i = route_i[:pos] + route_i[pos+1:]
                    d_i_new = route_distance(new_i)
                    
                    for ins_pos in range(len(route_j) + 1):
                        candidate_j = route_j[:ins_pos] + [cust] + route_j[ins_pos:]
                        d_j = route_distance(candidate_j)
                        if d_j < best_d_j:
                            best_d_j = d_j
                            best_new_j = candidate_j
                    
                    if best_new_j is None:
                        continue
                    
                    # Check feasibility
                    if not is_feasible(best_new_j) or not is_feasible(new_i):
                        continue
                    
                    total_new = d_i_new + best_d_j
                    total_old = route_distance(route_i) + route_distance(route_j)
                    
                    if total_new < total_old - 1e-9:
                        best_routes[i] = new_i
                        best_routes[j] = best_new_j
                        current_dist = sum(route_distance(r) for r in best_routes)
                        improved = True
                        break
        
        if improved:
            continue
        
        # Swap moves
        for i in range(len(best_routes)):
            if time.time() > deadline - 10:
                break
            route_i = best_routes[i]
            for pos_i in range(len(route_i)):
                if time.time() > deadline - 10:
                    break
                for j in range(i + 1, len(best_routes)):
                    if time.time() > deadline - 10:
                        break
                    route_j = best_routes[j]
                    for pos_j in range(len(route_j)):
                        if time.time() > deadline - 10:
                            break
                        
                        c_i = route_i[pos_i]
                        c_j = route_j[pos_j]
                        
                        new_i = route_i[:pos_i] + [c_j] + route_i[pos_i+1:]
                        new_j = route_j[:pos_j] + [c_i] + route_j[pos_j+1:]
                        
                        if not is_feasible(new_i) or not is_feasible(new_j):
                            continue
                        
                        total_new = route_distance(new_i) + route_distance(new_j)
                        total_old = route_distance(route_i) + route_distance(route_j)
                        
                        if total_new < total_old - 1e-9:
                            best_routes[i] = new_i
                            best_routes[j] = new_j
                            current_dist = sum(route_distance(r) for r in best_routes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        # Intra-route 2-opt refinement on each route
        if not improved:
            for i in range(len(best_routes)):
                if time.time() > deadline - 10:
                    break
                old_route = best_routes[i]
                if len(old_route) < 2:
                    continue
                new_route = optimize_route(old_route, max_iters=100)
                if route_distance(new_route) < route_distance(old_route) - 1e-9:
                    if is_feasible(new_route):
                        best_routes[i] = new_route
                        current_dist = sum(route_distance(r) for r in best_routes)
                        improved = True

    # Final validation and cleanup
    # Remove empty routes
    best_routes = [r for r in best_routes if r]
    
    # Ensure all customers are visited exactly once
    visited = set()
    for r in best_routes:
        for c in r:
            if c in visited:
                # Remove duplicate
                best_routes = [[x for x in route if x != c] if route is r else route for route in best_routes]
            visited.add(c)
    
    # Add any missing customers as single-customer routes
    for c in customers:
        if c not in visited:
            best_routes.append([c])
    
    return best_routes
