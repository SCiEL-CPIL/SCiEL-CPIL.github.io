"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random

def _build_dist_matrix(coords):
    n = len(coords)
    xs = np.array([c[0] for c in coords])
    ys = np.array([c[1] for c in coords])
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    return dist

def _check_route(route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon):
    """Check if a route is feasible. Returns (feasible, total_distance, total_time)."""
    if not route:
        return True, 0.0, 0.0
    
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

def _fast_check_route(route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon):
    """Fast feasibility check with early exits. Returns (feasible, total_distance)."""
    if not route:
        return True, 0.0
    
    total_demand = 0
    for v in route:
        if v <= n_customers:
            total_demand += demands[v]
    if total_demand > capacity:
        return False, 0.0
    
    battery = battery_max
    time_t = 0.0
    total_dist = 0.0
    prev = 0
    
    for v in route:
        d = dist[prev][v]
        battery -= d
        total_dist += d
        time_t += d / speed
        
        if battery < -1e-9:
            return False, 0.0
        
        if v <= n_customers:
            if time_t < ready[v]:
                time_t = ready[v]
            if time_t > due[v] + 1e-9:
                return False, 0.0
            time_t += service[v]
        else:
            energy_needed = battery_max - battery
            time_t += energy_needed * recharge_rate
            battery = battery_max
        
        prev = v
    
    d = dist[prev][0]
    battery -= d
    total_dist += d
    time_t += d / speed
    
    if battery < -1e-9 or time_t > horizon + 1e-9:
        return False, 0.0
    
    return True, total_dist

def _construct_initial_routes(instance, dist):
    """Construct initial routes using a greedy insertion heuristic."""
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
    
    routes = []
    route_demands = []
    unserved = list(range(1, n_customers + 1))
    
    # Sort customers by distance from depot to prioritize far ones or use a simple heuristic
    # Actually, let's just pick the one with largest demand first to fill capacity, or closest to depot?
    # Let's try sorting by distance from depot descending to start routes with far nodes?
    # Or just random/shuffled to break symmetry.
    random.shuffle(unserved)
    
    while unserved:
        # Start a new route
        route = []
        current_demand = 0
        
        # Find the best customer to start the route
        # Heuristic: Pick the one with largest demand that fits, or closest to depot?
        # Let's pick the one with largest demand to maximize capacity usage
        best_start = None
        max_demand = -1
        for c in unserved:
            if demands[c] <= capacity and demands[c] > max_demand:
                max_demand = demands[c]
                best_start = c
        
        if best_start is None:
             # If no single customer fits (shouldn't happen with demand 5-25 and cap 100), pick first
             best_start = unserved[0]
        
        route.append(best_start)
        current_demand = demands[best_start]
        unserved.remove(best_start)
        
        # Insert remaining customers into this route greedily
        while unserved:
            best_c = None
            best_pos = -1
            best_cost = float('inf')
            
            for c in unserved:
                if current_demand + demands[c] > capacity:
                    continue
                
                # Try inserting c at every position in route
                # Optimization: We can limit positions to neighbors of existing nodes or ends
                # For now, check all positions for small routes
                for pos in range(len(route) + 1):
                    # Quick distance check before full feasibility
                    # Calculate distance increase
                    if pos == 0:
                        if len(route) == 0:
                            inc = dist[0][c] + dist[c][0]
                        else:
                            inc = dist[0][c] + dist[c][route[0]] - dist[0][route[0]]
                    elif pos == len(route):
                        inc = dist[route[-1]][c] + dist[c][0] - dist[route[-1]][0]
                    else:
                        inc = dist[route[pos-1]][c] + dist[c][route[pos]] - dist[route[pos-1]][route[pos]]
                    
                    if inc < best_cost:
                        # Check if this insertion is feasible
                        new_route = route[:pos] + [c] + route[pos:]
                        feasible, _ = _fast_check_route(new_route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
                        if feasible:
                            best_cost = inc
                            best_c = c
                            best_pos = pos
            
            if best_c is not None:
                route = route[:best_pos] + [best_c] + route[best_pos:]
                current_demand += demands[best_c]
                unserved.remove(best_c)
            else:
                # No more customers can be added to this route
                break
        
        if route:
            routes.append(route)
            route_demands.append(current_demand)
    
    return routes

def _two_opt_improve(route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon, deadline):
    """Apply 2-opt improvements to a route."""
    if len(route) < 2:
        return route
    
    n = len(route)
    improved = True
    max_passes = 10
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
                prev_i = route[i-1] if i > 0 else 0
                next_j = route[j+1] if j < n - 1 else 0
                
                d_old = dist[prev_i][route[i]] + dist[route[j]][next_j]
                d_new = dist[prev_i][route[j]] + dist[route[i]][next_j]
                
                if d_new < d_old - 1e-9:
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    feasible, _ = _fast_check_route(new_route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
                    if feasible:
                        route = new_route
                        improved = True
                        break
            if improved:
                break
    
    return route

def _relocate_improve(route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon, deadline):
    """Apply Relocate (move 1 node) improvements."""
    if len(route) < 2:
        return route
    
    n = len(route)
    improved = True
    max_passes = 10
    passes = 0
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n):
            if time.time() > deadline:
                return route
            node = route[i]
            # Remove node
            rest = route[:i] + route[i+1:]
            
            # Try inserting at all other positions
            best_pos = -1
            best_cost = float('inf')
            
            # Calculate cost of removing node (approximate)
            # We just check all insertions in 'rest'
            for pos in range(len(rest) + 1):
                # Avoid inserting at same position (relative to original)
                # If pos == i, it's the same location. If pos == i-1 (and i>0) it's same.
                # Actually, just check distance improvement.
                
                new_route = rest[:pos] + [node] + rest[pos:]
                
                # Quick distance check
                if len(new_route) == 0: continue
                
                # Calculate distance delta
                # Original segment around i:
                if i == 0:
                    prev_i = 0
                    next_i = route[1] if n > 1 else 0
                    d_old_remove = dist[prev_i][node] + dist[node][next_i] - dist[prev_i][next_i]
                elif i == n-1:
                    prev_i = route[i-1]
                    next_i = 0
                    d_old_remove = dist[prev_i][node] + dist[node][next_i] - dist[prev_i][next_i]
                else:
                    prev_i = route[i-1]
                    next_i = route[i+1]
                    d_old_remove = dist[prev_i][node] + dist[node][next_i] - dist[prev_i][next_i]
                
                # New segment around pos:
                if pos == 0:
                    if len(rest) == 0:
                        d_new_insert = dist[0][node] + dist[node][0]
                    else:
                        d_new_insert = dist[0][node] + dist[node][rest[0]] - dist[0][rest[0]]
                elif pos == len(rest):
                    d_new_insert = dist[rest[-1]][node] + dist[node][0] - dist[rest[-1]][0]
                else:
                    d_new_insert = dist[rest[pos-1]][node] + dist[node][rest[pos]] - dist[rest[pos-1]][rest[pos]]
                
                total_delta = d_new_insert - d_old_remove
                
                if total_delta < -1e-9:
                    feasible, _ = _fast_check_route(new_route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
                    if feasible and total_delta < best_cost:
                        best_cost = total_delta
                        best_pos = pos
            
            if best_pos >= 0:
                rest = route[:i] + route[i+1:]
                route = rest[:best_pos] + [node] + rest[best_pos:]
                improved = True
                break # Restart loop
    
    return route

def _solve_instance(instance, time_limit=45):
    """Solve a single instance with time limit."""
    start_time = time.time()
    deadline = start_time + time_limit
    
    coords = instance["coords"]
    dist = _build_dist_matrix(coords)
    
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
    
    routes = _construct_initial_routes(instance, dist)
    
    # Local Search Loop
    while time.time() < deadline - 1:
        for i in range(len(routes)):
            if time.time() > deadline - 1:
                break
            # Alternate between 2-opt and Relocate
            if i % 2 == 0:
                routes[i] = _two_opt_improve(routes[i], dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon, deadline - 1)
            else:
                routes[i] = _relocate_improve(routes[i], dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon, deadline - 1)
        
        # Check if we have time for another pass
        if time.time() > deadline - 5:
            break

    # Final verification
    valid_routes = []
    for r in routes:
        feasible, _, _ = _check_route(r, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
        if feasible:
            valid_routes.append(r)
    
    all_served = set()
    for r in valid_routes:
        all_served.update(r)
    
    missing = [c for c in range(1, n_customers + 1) if c not in all_served]
    
    for c in missing:
        valid_routes.append([c])
    
    return valid_routes

def plan(instance):
    """Main entry point. Returns routes for the instance."""
    start_time = time.time()
    per_instance_time = 45
    
    routes = _solve_instance(instance, per_instance_time)
    
    return routes
# EVOLVE-BLOCK-END
