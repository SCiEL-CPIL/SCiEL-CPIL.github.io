# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def _dist_matrix(coords):
    n = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            d = math.sqrt(dx * dx + dy * dy)
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _check_route(route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon):
    """Check if a route is feasible. Returns (feasible, total_distance, total_time)."""
    if not route:
        return True, 0.0, 0.0
    
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    
    total_demand = sum(demands[v] for v in route if v <= n_customers)
    if total_demand > capacity:
        return False, 0.0, 0.0
    
    battery = battery_max
    time_t = 0.0
    total_dist = 0.0
    prev = 0  # depot
    
    for v in route:
        d = dist[prev][v]
        battery -= d
        time_t += d / speed
        total_dist += d
        
        if battery < -1e-9:
            return False, 0.0, 0.0
        
        if v <= n_customers:
            if time_t < ready[v]:
                time_t = ready[v]
            if time_t > due[v] + 1e-9:
                return False, 0.0, 0.0
            time_t += service[v]
        else:
            energy_needed = battery_max - battery
            time_t += energy_needed * recharge_rate
            battery = battery_max
        
        prev = v
    
    d = dist[prev][0]
    battery -= d
    time_t += d / speed
    total_dist += d
    
    if battery < -1e-9:
        return False, 0.0, 0.0
    if time_t > horizon + 1e-9:
        return False, 0.0, 0.0
    
    return True, total_dist, time_t


def _route_distance(route, dist):
    """Quick distance calculation for a route."""
    if not route:
        return 0.0
    total = 0.0
    prev = 0
    for v in route:
        total += dist[prev][v]
        prev = v
    total += dist[prev][0]
    return total


def _cluster_customers(coords, n_customers, n_clusters=5):
    """Simple k-means-like clustering to group spatially close customers."""
    customers = list(range(1, n_customers + 1))
    random.shuffle(customers)
    
    # Initialize centroids
    centroids = [np.array(coords[c], dtype=float) for c in customers[:n_clusters]]
    assignments = {}
    
    for _ in range(10):
        new_assignments = {}
        for c in customers:
            best_cluster = 0
            best_dist = float('inf')
            for i, cent in enumerate(centroids):
                d = np.linalg.norm(np.array(coords[c]) - cent)
                if d < best_dist:
                    best_dist = d
                    best_cluster = i
            new_assignments[c] = best_cluster
        
        if new_assignments == assignments:
            break
        assignments = new_assignments
        
        # Update centroids
        for i in range(n_clusters):
            members = [c for c in customers if assignments[c] == i]
            if members:
                centroids[i] = np.mean([np.array(coords[c]) for c in members], axis=0)
    
    clusters = [[] for _ in range(n_clusters)]
    for c in customers:
        clusters[assignments[c]].append(c)
    
    return clusters


def _construct_cluster_routes(instance, dist, time_limit=5):
    """Construct initial routes using spatial clustering + time-window-aware insertion."""
    start_time = time.time()
    deadline = start_time + time_limit
    
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    battery_max = instance["battery"]
    capacity = instance["capacity"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    # Cluster customers spatially
    n_clusters = min(5, n_customers)
    clusters = _cluster_customers(coords, n_customers, n_clusters)
    
    # Sort clusters by distance from depot to process closer ones first? 
    # Or just process all. Let's sort customers within clusters by due date.
    
    routes = []
    
    for cluster in clusters:
        if not cluster:
            continue
        
        # Sort by due date within cluster
        cluster.sort(key=lambda c: due[c])
        
        current_route = []
        current_demand = 0
        
        for c in cluster:
            if time.time() > deadline:
                break
            
            if current_demand + demands[c] > capacity:
                if current_route:
                    routes.append(current_route)
                    current_route = []
                    current_demand = 0
                else:
                    routes.append([c])
                    continue
            
            # Try to insert c into current_route at best feasible position
            best_pos = -1
            best_cost = float('inf')
            
            for pos in range(len(current_route) + 1):
                new_route = current_route[:pos] + [c] + current_route[pos:]
                feasible, d, t = _check_route(new_route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                if feasible and d < best_cost:
                    best_cost = d
                    best_pos = pos
            
            if best_pos != -1:
                current_route = current_route[:best_pos] + [c] + current_route[best_pos:]
                current_demand += demands[c]
            else:
                # Can't insert, start new route with just this customer
                if current_route:
                    routes.append(current_route)
                current_route = [c]
                current_demand = demands[c]
        
        if current_route:
            routes.append(current_route)
    
    # Check for any missing customers
    all_served = set()
    for r in routes:
        all_served.update(r)
    missing = [c for c in range(1, n_customers + 1) if c not in all_served]
    for c in missing:
        routes.append([c])
    
    return routes


def _local_search(routes, dist, instance, time_limit=40):
    """
    Efficient local search with 2-Opt, Or-Opt, and Relocate moves.
    Uses fast distance delta calculations to avoid full route re-evaluation
    unless a potential improvement is found.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    n_customers = instance["n_customers"]
    battery_max = instance["battery"]
    capacity = instance["capacity"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    # Precompute route distances
    route_dists = [_route_distance(r, dist) for r in routes]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. 2-Opt within routes
        for i in range(len(routes)):
            r = routes[i]
            n = len(r)
            if n < 2:
                continue
            
            # Skip 2-opt for very small routes as it's less likely to help or is trivial
            for a in range(n - 1):
                for b in range(a + 1, n):
                    # Calculate delta distance for reversing r[a:b+1]
                    # Old edges: (r[a-1], r[a]) and (r[b], r[b+1])
                    # New edges: (r[a-1], r[b]) and (r[a], r[b+1])
                    # Note: depot is index 0.
                    
                    prev_a = r[a-1] if a > 0 else 0
                    next_b = r[b+1] if b < n - 1 else 0
                    
                    old_d = dist[prev_a][r[a]] + dist[r[b]][next_b]
                    new_d = dist[prev_a][r[b]] + dist[r[a]][next_b]
                    
                    if new_d < old_d - 1e-9:
                        new_r = r[:a] + r[a:b+1][::-1] + r[b+1:]
                        # Check feasibility
                        feasible, d, t = _check_route(new_r, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                        if feasible:
                            routes[i] = new_r
                            route_dists[i] = d
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            continue
            
        # 2. Or-Opt within routes (move 1 or 2 consecutive nodes)
        for i in range(len(routes)):
            r = routes[i]
            n = len(r)
            if n < 3:
                continue
            
            for L in [1, 2]:
                if n < L + 1:
                    continue
                for start in range(n - L):
                    moving = r[start:start+L]
                    # Only move customers? Or stations too? 
                    # Stations are tricky in Or-Opt because they change battery state.
                    # To be safe, only move customers in Or-Opt, or ensure stations are handled.
                    # Given the complexity, let's only move customers for Or-Opt.
                    if any(m > n_customers for m in moving):
                        continue
                    
                    rest = r[:start] + r[start+L:]
                    best_pos = -1
                    best_delta = float('inf')
                    
                    # Try inserting 'moving' into 'rest'
                    # We need to check feasibility.
                    # Fast distance check:
                    # Old edges around start: (r[start-1], r[start]) and (r[start+L-1], r[start+L])
                    # If start==0, prev is depot. If start+L==n, next is depot.
                    
                    for pos in range(len(rest) + 1):
                        # Skip original position
                        if pos == start or (start > 0 and pos == start + L) or (start == 0 and pos == L):
                            continue
                        
                        # Calculate new distance
                        # This is complex because 'rest' changes.
                        # Let's just compute distance of new route if it looks promising.
                        # Actually, let's just check feasibility and distance for valid positions.
                        new_r = rest[:pos] + moving + rest[pos:]
                        
                        # Quick distance check
                        d_new = _route_distance(new_r, dist)
                        if d_new < route_dists[i] - 1e-9:
                            feasible, d, t = _check_route(new_r, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                            if feasible:
                                best_pos = pos
                                best_delta = d_new
                                # Don't break, might find better position
                                # But for speed, we can break on first improvement? 
                                # No, let's find best improvement in this iteration.
                                # To save time, break on first feasible improvement.
                                break
                    
                    if best_pos != -1:
                        new_r = rest[:best_pos] + moving + rest[best_pos:]
                        routes[i] = new_r
                        route_dists[i] = best_delta
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            continue

        # 3. Relocate: Move customer from route i to route j
        for i in range(len(routes)):
            r_i = routes[i]
            if not r_i:
                continue
            for pos_i in range(len(r_i)):
                c = r_i[pos_i]
                if c > n_customers:
                    continue
                
                # Try moving c to other routes
                for j in range(len(routes)):
                    if i == j:
                        continue
                    r_j = routes[j]
                    
                    # Check capacity
                    demand_i = sum(instance["demands"][v] for v in r_i if v <= n_customers)
                    demand_j = sum(instance["demands"][v] for v in r_j if v <= n_customers)
                    
                    if demand_j + instance["demands"][c] > capacity:
                        continue
                    if demand_i - instance["demands"][c] < 0:
                        continue # Should not happen
                    
                    best_pos_j = -1
                    best_dist_j = float('inf')
                    
                    for pos_j in range(len(r_j) + 1):
                        new_r_j = r_j[:pos_j] + [c] + r_j[pos_j:]
                        
                        # Quick distance check
                        d_j_new = _route_distance(new_r_j, dist)
                        if d_j_new < best_dist_j:
                            # Check feasibility
                            feasible, d, t = _check_route(new_r_j, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                            if feasible:
                                best_dist_j = d
                                best_pos_j = pos_j
                                # Break on first feasible? Or find best?
                                # Let's find best to improve quality.
                                # But for speed, we can break.
                                # Let's break to save time.
                                # Actually, let's just check a few positions.
                                # To be safe and fast, break on first feasible improvement over current?
                                # No, we need to compare total distance.
                                # Let's just find the best feasible position in r_j.
                    
                    if best_pos_j != -1:
                        new_r_j = r_j[:best_pos_j] + [c] + r_j[best_pos_j:]
                        new_r_i = r_i[:pos_i] + r_i[pos_i+1:]
                        
                        d_i_new = _route_distance(new_r_i, dist) if new_r_i else 0.0
                        d_j_new = best_dist_j
                        
                        total_new = d_i_new + d_j_new
                        total_old = route_dists[i] + route_dists[j]
                        
                        if total_new < total_old - 1e-9:
                            # Check feasibility of new_r_i
                            if new_r_i:
                                feasible_i, _, _ = _check_route(new_r_i, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                            else:
                                feasible_i = True
                            
                            if feasible_i:
                                routes[i] = new_r_i
                                routes[j] = new_r_j
                                route_dists[i] = d_i_new
                                route_dists[j] = d_j_new
                                improved = True
                                break
                    if improved:
                        break
            if improved:
                break
            
            # Remove empty routes
            if not r_i:
                routes.pop(i)
                route_dists.pop(i)
                break # Indices changed, restart loop
        
        if improved:
            # Restart loop to check for new improvements
            continue
            
    return routes


def _solve_instance(instance, time_limit=45):
    """Solve a single instance."""
    start_time = time.time()
    deadline = start_time + time_limit
    
    coords = instance["coords"]
    dist = _dist_matrix(coords)
    
    battery_max = instance["battery"]
    capacity = instance["capacity"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    n_customers = instance["n_customers"]
    
    # 1. Construct initial routes using clustering
    construct_time = min(3, time_limit * 0.1)
    routes = _construct_cluster_routes(instance, dist, construct_time)
    
    # Ensure all customers are served
    all_served = set()
    for r in routes:
        all_served.update(r)
    missing = [c for c in range(1, n_customers + 1) if c not in all_served]
    for c in missing:
        routes.append([c])
    
    # 2. Local Search
    ls_time = deadline - time.time() - 2
    if ls_time > 1:
        routes = _local_search(routes, dist, instance, ls_time)
    
    # Final verification
    valid_routes = []
    for r in routes:
        feasible, _, _ = _check_route(r, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
        if feasible:
            valid_routes.append(r)
    
    # Check all customers are served
    all_served = set()
    for r in valid_routes:
        all_served.update(r)
    
    missing = [c for c in range(1, n_customers + 1) if c not in all_served]
    for c in missing:
        valid_routes.append([c])
    
    return valid_routes


def plan(instance):
    """Main entry point."""
    start_time = time.time()
    
    # 3 instances, 150s total, leave 15s buffer -> 45s per instance
    per_instance_time = 45
    
    routes = _solve_instance(instance, per_instance_time)
    
    return routes
# EVOLVE-BLOCK-END
