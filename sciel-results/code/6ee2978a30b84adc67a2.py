"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def _dist(coords, i, j):
    dx = coords[i][0] - coords[j][0]
    dy = coords[i][1] - coords[j][1]
    return math.sqrt(dx * dx + dy * dy)


def _build_dist_matrix(coords):
    n = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _dist(coords, i, j)
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _check_route(route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon):
    """Check if a route is feasible. Returns (feasible, total_distance, total_time)."""
    if not route:
        return True, 0.0, 0.0
    
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = set(instance["stations"])
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
            # Customer: wait if early
            if time_t < ready[v]:
                time_t = ready[v]
            if time_t > due[v] + 1e-9:
                return False, 0.0, 0.0
            time_t += service[v]
        else:
            # Charging station
            energy_needed = battery_max - battery
            time_t += energy_needed * recharge_rate
            battery = battery_max
        
        prev = v
    
    # Return to depot
    d = dist[prev][0]
    battery -= d
    time_t += d / speed
    total_dist += d
    
    if battery < -1e-9:
        return False, 0.0, 0.0
    if time_t > horizon + 1e-9:
        return False, 0.0, 0.0
    
    return True, total_dist, time_t


def _construct_initial_routes(instance, dist):
    """Construct initial routes using a savings-based heuristic."""
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
    stations = instance["stations"]
    
    # Start with each customer as its own route
    routes = [[i] for i in range(1, n_customers + 1)]
    
    # Calculate savings for merging routes
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
    
    # Try to merge routes
    for s, ci, cj in savings:
        ri = route_of.get(ci, -1)
        rj = route_of.get(cj, -1)
        if ri == -1 or rj == -1 or ri == rj:
            continue
        
        # Try both orderings: [route_i, route_j] and [route_j, route_i]
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
            for c in best_new_route:
                route_of[c] = ri
    
    # Remove empty routes
    routes = [r for r in routes if r]
    
    # Insert any remaining customers (shouldn't be any, but just in case)
    all_served = set()
    for r in routes:
        all_served.update(r)
    
    for c in range(1, n_customers + 1):
        if c not in all_served:
            # Try to insert into existing route
            inserted = False
            for idx, r in enumerate(routes):
                # Simple insertion: try all positions
                best_pos = -1
                best_cost = float('inf')
                current_demand = sum(demands[v] for v in r if v <= n_customers)
                if current_demand + demands[c] > capacity:
                    continue
                for pos in range(len(r) + 1):
                    new_route = r[:pos] + [c] + r[pos:]
                    feasible, total_dist, _ = _check_route(new_route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                    if feasible:
                        # Approximate cost increase
                        if pos == 0:
                            inc = dist[0][c] + dist[c][r[0]] - dist[0][r[0]]
                        elif pos == len(r):
                            inc = dist[r[-1]][c] + dist[c][0] - dist[r[-1]][0]
                        else:
                            inc = dist[r[pos-1]][c] + dist[c][r[pos]] - dist[r[pos-1]][r[pos]]
                        if inc < best_cost:
                            best_cost = inc
                            best_pos = pos
                if best_pos >= 0:
                    routes[idx] = r[:best_pos] + [c] + r[best_pos:]
                    inserted = True
                    break
            if not inserted:
                routes.append([c])
    
    return routes


def _two_opt_improve(route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon, deadline):
    """Apply 2-opt improvements to a route. Returns improved route."""
    if len(route) < 2:
        return route
    
    n = len(route)
    improved = True
    max_passes = 10 # Limit passes to avoid infinite loops
    passes = 0
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n - 1):
            if time.time() > deadline:
                return route
            for j in range(i + 1, n):
                if time.time() > deadline:
                    return route
                # Reverse segment i..j
                prev_i = route[i-1] if i > 0 else 0
                next_j = route[j+1] if j < n - 1 else 0
                
                # Calculate distance difference
                d_old = dist[prev_i][route[i]] + dist[route[j]][next_j]
                d_new = dist[prev_i][route[j]] + dist[route[i]][next_j]
                
                if d_new < d_old - 1e-9:
                    # Potential improvement, check feasibility
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    feasible, _, _ = _check_route(new_route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                    if feasible:
                        route = new_route
                        improved = True
                        break
            if improved:
                break
    
    return route


def _solve_instance(instance, time_limit=30):
    """Solve a single instance with time limit."""
    start_time = time.time()
    deadline = start_time + time_limit
    
    coords = instance["coords"]
    dist = _build_dist_matrix(coords)
    
    battery_max = instance["battery"]
    capacity = instance["capacity"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    # Construct initial routes
    routes = _construct_initial_routes(instance, dist)
    
    # Local search loop: Just 2-opt for now to ensure speed
    while time.time() < deadline - 2: # Leave 2s buffer
        for i in range(len(routes)):
            if time.time() > deadline - 2:
                break
            routes[i] = _two_opt_improve(routes[i], dist, instance, battery_max, capacity, speed, recharge_rate, horizon, deadline - 2)
        
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
    """Main entry point. Returns routes for the instance."""
    start_time = time.time()
    # Leave 10s safety margin for 3 instances. 120s total limit.
    # 3 instances * 30s = 90s. 30s buffer.
    per_instance_time = 30
    
    routes = _solve_instance(instance, per_instance_time)
    
    return routes
# EVOLVE-BLOCK-END
