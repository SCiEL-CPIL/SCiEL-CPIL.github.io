"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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
    
    total_demand = 0
    for v in route:
        if v <= n_customers:
            total_demand += demands[v]
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


def _nearest_neighbor_route(customers, dist, instance, battery_max, capacity, speed, recharge_rate, horizon):
    """Build a single route using nearest neighbor heuristic."""
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    
    route = []
    unvisited = list(customers)
    current_pos = 0
    battery = battery_max
    time_t = 0.0
    total_dist = 0.0
    total_demand = 0.0
    
    while unvisited:
        # Find nearest customer that fits
        best_customer = None
        best_dist = float('inf')
        for c in unvisited:
            if total_demand + demands[c] > capacity:
                continue
            d = dist[current_pos][c]
            if battery - d < -1e-9:
                continue
            if d < best_dist:
                best_dist = d
                best_customer = c
        
        if best_customer is None:
            break
        
        # Go to customer
        d = dist[current_pos][best_customer]
        battery -= d
        time_t += d / speed
        total_dist += d
        
        # Check time window
        if time_t < ready[best_customer]:
            time_t = ready[best_customer]
        if time_t > due[best_customer] + 1e-9:
            # Can't serve in time, skip
            unvisited.remove(best_customer)
            continue
        
        time_t += service[best_customer]
        route.append(best_customer)
        unvisited.remove(best_customer)
        total_demand += demands[best_customer]
        current_pos = best_customer
    
    return route


def _construct_clustering_routes(instance, dist):
    """Construct routes using a clustering approach based on angular proximity."""
    coords = instance["coords"]
    demands = instance["demands"]
    n_customers = instance["n_customers"]
    battery_max = instance["battery"]
    capacity = instance["capacity"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    # Compute angle from depot for each customer
    depot_x, depot_y = coords[0]
    angles = []
    for i in range(1, n_customers + 1):
        dx = coords[i][0] - depot_x
        dy = coords[i][1] - depot_y
        angle = math.atan2(dy, dx)
        angles.append((angle, i))
    
    # Sort by angle
    angles.sort()
    
    # Group customers into clusters based on angular proximity
    routes = []
    current_cluster = []
    current_demand = 0
    
    for angle, customer in angles:
        if current_demand + demands[customer] <= capacity:
            current_cluster.append(customer)
            current_demand += demands[customer]
        else:
            if current_cluster:
                routes.append(current_cluster)
            current_cluster = [customer]
            current_demand = demands[customer]
    
    if current_cluster:
        routes.append(current_cluster)
    
    return routes


def _solve_with_savings(instance, dist, time_limit=20):
    """Solve using Clarke-Wright savings heuristic."""
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
    
    # Start with each customer as its own route
    routes = [[i] for i in range(1, n_customers + 1)]
    
    # Calculate savings
    savings = []
    for i in range(n_customers):
        for j in range(i + 1, n_customers):
            ci = i + 1
            cj = j + 1
            s = dist[0][ci] + dist[0][cj] - dist[ci][cj]
            savings.append((s, ci, cj))
    
    # Sort by savings descending
    savings.sort(reverse=True)
    
    # Track which route each customer belongs to
    route_of = {}
    for idx, r in enumerate(routes):
        for c in r:
            route_of[c] = idx
    
    # Precompute route demands
    route_demand = [demands[r[0]] for r in routes]
    
    # Try to merge routes
    for s, ci, cj in savings:
        if time.time() > deadline - 2:
            break
        
        ri = route_of.get(ci, -1)
        rj = route_of.get(cj, -1)
        if ri == -1 or rj == -1 or ri == rj:
            continue
        
        # Check capacity
        if route_demand[ri] + route_demand[rj] > capacity:
            continue
        
        # Try both orderings
        best_new_route = None
        best_dist = float('inf')
        
        for order in [[routes[ri], routes[rj]], [routes[rj], routes[ri]]]:
            merged = order[0] + order[1]
            feasible, total_dist, _ = _check_route(merged, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
            if feasible and total_dist < best_dist:
                best_dist = total_dist
                best_new_route = merged
        
        if best_new_route is not None:
            routes[ri] = best_new_route
            routes[rj] = []
            route_demand[ri] += route_demand[rj]
            route_demand[rj] = 0
            for c in best_new_route:
                route_of[c] = ri
    
    # Remove empty routes
    routes = [r for r in routes if r]
    
    return routes


def _two_opt_improve(route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon, deadline):
    """Apply 2-opt improvements to a route."""
    if len(route) < 2:
        return route
    
    n = len(route)
    improved = True
    max_passes = 10
    
    while improved and max_passes > 0:
        improved = False
        max_passes -= 1
        for i in range(n - 1):
            if time.time() > deadline:
                return route
            for j in range(i + 1, n):
                if time.time() > deadline:
                    return route
                prev_i = route[i-1] if i > 0 else 0
                next_j = route[j+1] if j < n - 1 else 0
                
                d_old = dist[prev_i][route[i]] + dist[route[j]][next_j]
                d_new = dist[prev_i][route[j]] + dist[route[i]][next_j]
                
                if d_new < d_old - 1e-9:
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    feasible, _, _ = _check_route(new_route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                    if feasible:
                        route = new_route
                        n = len(route)
                        improved = True
                        break
            if improved:
                break
    
    return route


def _or_opt_improve(route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon, deadline):
    """Apply Or-opt improvements to a route."""
    if len(route) < 3:
        return route
    
    n = len(route)
    improved = True
    max_passes = 5
    
    while improved and max_passes > 0:
        improved = False
        max_passes -= 1
        for k in range(1, min(3, n)):
            for i in range(n - k):
                if time.time() > deadline:
                    return route
                # Extract segment of length k starting at i
                segment = route[i:i+k]
                
                # Try inserting at different positions
                for j in range(n - k):
                    if j == i:
                        continue
                    if j >= i and j < i + k:
                        continue
                    
                    # Compute new route
                    new_route = route[:i] + route[i+k:]
                    insert_pos = j if j < i else j - k
                    new_route = new_route[:insert_pos] + segment + new_route[insert_pos:]
                    
                    # Check if it's different from original
                    if new_route == route:
                        continue
                    
                    feasible, _, _ = _check_route(new_route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                    if feasible:
                        d_old = 0
                        for idx in range(len(route)):
                            prev = route[idx-1] if idx > 0 else 0
                            d_old += dist[prev][route[idx]]
                        d_old += dist[route[-1]][0]
                        
                        d_new = 0
                        for idx in range(len(new_route)):
                            prev = new_route[idx-1] if idx > 0 else 0
                            d_new += dist[prev][new_route[idx]]
                        d_new += dist[new_route[-1]][0]
                        
                        if d_new < d_old - 1e-9:
                            route = new_route
                            n = len(route)
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    
    return route


def _solve_instance(instance, time_limit=25):
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
    
    # Use savings heuristic
    routes = _solve_with_savings(instance, dist, time_limit - 10)
    
    # Apply 2-opt and Or-opt local search
    while time.time() < deadline - 3:
        for i in range(len(routes)):
            if time.time() > deadline - 3:
                break
            routes[i] = _two_opt_improve(routes[i], dist, instance, battery_max, capacity, speed, recharge_rate, horizon, deadline - 3)
            routes[i] = _or_opt_improve(routes[i], dist, instance, battery_max, capacity, speed, recharge_rate, horizon, deadline - 3)
    
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
    
    n_customers = instance["n_customers"]
    missing = [c for c in range(1, n_customers + 1) if c not in all_served]
    
    for c in missing:
        valid_routes.append([c])
    
    return valid_routes


def plan(instance):
    """Main entry point."""
    start_time = time.time()
    
    # 3 instances, 150s total, leave 10s buffer -> 40s per instance
    per_instance_time = 40
    
    routes = _solve_instance(instance, per_instance_time)
    
    return routes
# EVOLVE-BLOCK-END
