"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import sqrt

def _build_distance_matrix(coords):
    """Precompute Euclidean distance matrix using NumPy."""
    coords = np.array(coords, dtype=np.float64)
    # Compute pairwise distances
    x_diff = coords[:, 0, np.newaxis] - coords[np.newaxis, :, 0]
    y_diff = coords[:, 1, np.newaxis] - coords[np.newaxis, :, 1]
    dist_matrix = np.sqrt(x_diff**2 + y_diff**2)
    return dist_matrix

def _simulate_route(route, dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon):
    """
    Simulate a route and return (total_distance, is_feasible).
    Optimized for speed: uses local variables and minimal overhead.
    """
    if not route:
        return 0.0, True
        
    cur_node = 0
    cur_time = 0.0
    cur_bat = battery
    total_dist = 0.0
    load = 0.0
    
    # Local references for speed
    d_mat = dist_matrix
    dem = demands
    rd = ready
    du = due
    svc = service
    st_set = stations_set
    cap = capacity
    bat_max = battery
    spd = speed
    rc_rate = recharge_rate
    hor = horizon
    
    for node in route:
        d = d_mat[cur_node, node]
        cur_time += d / spd
        cur_bat -= d
        total_dist += d
        
        if cur_bat < -1e-9:
            return total_dist, False
        
        if node in st_set:
            # Recharge
            energy_needed = bat_max - cur_bat
            if energy_needed > 0:
                cur_time += energy_needed * rc_rate
                cur_bat = bat_max
        else:
            # Customer
            if cur_time < rd[node]:
                cur_time = rd[node]
            if cur_time > du[node] + 1e-9:
                return total_dist, False
            cur_time += svc[node]
            load += dem[node]
            if load > cap + 1e-9:
                return total_dist, False
        
        cur_node = node
    
    # Return to depot
    d = d_mat[cur_node, 0]
    cur_time += d / spd
    cur_bat -= d
    total_dist += d
    
    if cur_bat < -1e-9:
        return total_dist, False
    if cur_time > hor + 1e-9:
        return total_dist, False
        
    return total_dist, True

def _solve_instance(instance, time_limit=45.0):
    start_time = time.time()
    
    coords = instance["coords"]
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    stations_set = set(stations)
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    # Precompute distances
    dist_matrix = _build_distance_matrix(coords)
    
    customers = list(range(1, n_customers + 1))
    # Sort by due date for initial construction heuristic
    customers.sort(key=lambda v: due[v])
    
    routes = []
    
    # Helper to calculate route distance quickly
    def calc_route_dist(route):
        if not route:
            return 0.0
        d = dist_matrix[0, route[0]]
        for i in range(len(route) - 1):
            d += dist_matrix[route[i], route[i+1]]
        d += dist_matrix[route[-1], 0]
        return d

    # --- Construction Phase: Cheapest Insertion ---
    # Start with empty routes, insert customers one by one
    # To avoid O(N^3) in insertion check, we limit insertion attempts per customer
    # or use a simpler greedy if time is tight. 
    # Given N=25, O(N^3) is 15625 ops, which is trivial.
    
    for cust in customers:
        best_route_idx = -1
        best_pos = -1
        best_addl_dist = float('inf')
        
        # Try inserting into existing routes
        for ri, route in enumerate(routes):
            # Try inserting at each position
            for pos in range(len(route) + 1):
                # Quick check: if capacity exceeded, skip
                current_load = sum(demands[c] for c in route if c not in stations_set)
                if current_load + demands[cust] > capacity + 1e-9:
                    continue
                
                new_route = route[:pos] + [cust] + route[pos:]
                
                # Feasibility check
                _, feasible = _simulate_route(new_route, dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon)
                if not feasible:
                    continue
                
                # Calculate additional distance
                old_d = calc_route_dist(route)
                new_d = calc_route_dist(new_route)
                addl = new_d - old_d
                
                if addl < best_addl_dist:
                    best_addl_dist = addl
                    best_route_idx = ri
                    best_pos = pos
        
        if best_route_idx >= 0:
            routes[best_route_idx].insert(best_pos, cust)
        else:
            # Start new route
            # Check if single customer route is feasible
            _, feasible = _simulate_route([cust], dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon)
            if feasible:
                routes.append([cust])
            else:
                # Should not happen with given constraints, but handle gracefully
                # If even a single customer is infeasible, the instance might be unsolvable
                # or we need to insert a station. 
                # For now, we assume it's feasible or we skip (which would be invalid).
                # Let's try adding a station before it?
                # If this happens, the solution is likely invalid. 
                # We'll just add it and hope the repair phase fixes it or it's actually feasible via station.
                # Actually, _simulate_route checks full feasibility. If [cust] is infeasible, 
                # we can't just add it. 
                # However, battery=300 and max dist ~141, so [cust] should always be feasible.
                routes.append([cust])

    # --- Local Search Phase ---
    # We will perform multiple passes of 2-opt, Or-opt, and Relocation.
    # Stop when no improvement or time limit reached.
    
    current_total_dist = sum(calc_route_dist(r) for r in routes)
    best_routes = [r[:] for r in routes]
    best_dist = current_total_dist
    
    improved = True
    while improved and (time.time() - start_time < time_limit * 0.9):
        improved = False
        
        # 1. Intra-route 2-Opt
        for ri in range(len(routes)):
            route = routes[ri]
            n = len(route)
            if n < 2:
                continue
            
            for i in range(n - 1):
                for j in range(i + 1, n):
                    # Reverse segment i..j
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    
                    # Quick distance delta check
                    # Old dist: ... d(route[i-1], route[i]) + d(route[j], route[j+1]) ...
                    # New dist: ... d(route[i-1], route[j]) + d(route[i], route[j+1]) ...
                    # This is an approximation, full simulation is safer but slower.
                    # Given small N, full simulation is fine.
                    _, feasible = _simulate_route(new_route, dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon)
                    if not feasible:
                        continue
                    
                    new_d = calc_route_dist(new_route)
                    old_d = calc_route_dist(route)
                    
                    if new_d < old_d - 1e-6:
                        routes[ri] = new_route
                        current_total_dist += new_d - old_d
                        improved = True
                        # Break inner loops to restart search with new state
                        break
                if improved:
                    break
        
        if improved:
            if current_total_dist < best_dist - 1e-6:
                best_dist = current_total_dist
                best_routes = [r[:] for r in routes]
            continue # Restart loop to take advantage of changes
        
        # 2. Intra-route Or-opt (move segment of length 1, 2, 3)
        for ri in range(len(routes)):
            route = routes[ri]
            n = len(route)
            if n < 2:
                continue
            
            found_improvement = False
            for seg_len in [1, 2, 3]:
                for i in range(n - seg_len + 1):
                    seg = route[i:i+seg_len]
                    # Remove segment
                    rest = route[:i] + route[i+seg_len:]
                    if not rest:
                        continue
                        
                    # Try inserting segment at each position in rest
                    for pos in range(len(rest) + 1):
                        if pos == i or pos == i + seg_len:
                            continue # Same position or adjacent without change in order (mostly)
                        
                        new_route = rest[:pos] + seg + rest[pos:]
                        
                        _, feasible = _simulate_route(new_route, dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon)
                        if not feasible:
                            continue
                        
                        new_d = calc_route_dist(new_route)
                        old_d = calc_route_dist(route)
                        
                        if new_d < old_d - 1e-6:
                            routes[ri] = new_route
                            current_total_dist += new_d - old_d
                            found_improvement = True
                            improved = True
                            break
                    if found_improvement:
                        break
                if found_improvement:
                    break
            
            if found_improvement:
                if current_total_dist < best_dist - 1e-6:
                    best_dist = current_total_dist
                    best_routes = [r[:] for r in routes]
                break # Restart outer while loop
        
        if improved:
            continue
            
        # 3. Inter-route Relocation
        # Move a customer from route A to route B
        found_improvement = False
        for ri in range(len(routes)):
            for rj in range(len(routes)):
                if ri == rj:
                    continue
                
                route_i = routes[ri]
                route_j = routes[rj]
                
                if not route_i:
                    continue
                
                for pos_i in range(len(route_i)):
                    cust = route_i[pos_i]
                    if cust in stations_set:
                        continue
                    
                    # Remove from route_i
                    new_route_i = route_i[:pos_i] + route_i[pos_i+1:]
                    
                    # Check if route_i is still feasible (might not need to if we just remove, 
                    # but removing a customer might make a station visit unnecessary or change timing)
                    _, feasible_i = _simulate_route(new_route_i, dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon)
                    if not feasible_i:
                        continue
                    
                    # Try inserting into route_j
                    best_pos_j = -1
                    best_addl = float('inf')
                    base_d_j = calc_route_dist(route_j)
                    
                    for pos_j in range(len(route_j) + 1):
                        new_route_j = route_j[:pos_j] + [cust] + route_j[pos_j:]
                        
                        # Quick capacity check
                        load_j = sum(demands[c] for c in route_j if c not in stations_set)
                        if load_j + demands[cust] > capacity + 1e-9:
                            continue
                        
                        _, feasible_j = _simulate_route(new_route_j, dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon)
                        if not feasible_j:
                            continue
                        
                        new_d_j = calc_route_dist(new_route_j)
                        addl = new_d_j - base_d_j
                        
                        if addl < best_addl:
                            best_addl = addl
                            best_pos_j = pos_j
                    
                    if best_pos_j >= 0:
                        new_route_j = route_j[:best_pos_j] + [cust] + route_j[best_pos_j:]
                        new_d_i = calc_route_dist(new_route_i)
                        old_total = calc_route_dist(route_i) + calc_route_dist(route_j)
                        new_total = new_d_i + calc_route_dist(new_route_j)
                        
                        if new_total < old_total - 1e-6:
                            routes[ri] = new_route_i
                            routes[rj] = new_route_j
                            current_total_dist += new_total - old_total
                            found_improvement = True
                            improved = True
                            break
                if found_improvement:
                    break
            if found_improvement:
                if current_total_dist < best_dist - 1e-6:
                    best_dist = current_total_dist
                    best_routes = [r[:] for r in routes]
                break # Restart outer while loop

    # Return best found routes
    # Ensure all customers are served
    # The construction ensures all are served if feasible.
    # Local search preserves feasibility.
    
    return best_routes

def plan(instance):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    try:
        # Adjust time limit based on available time
        available_time = deadline - time.time()
        # Reserve 5s for final checks/return
        search_time = max(1.0, available_time - 5.0)
        
        routes = _solve_instance(instance, time_limit=search_time)
        
        # Final verification and repair if needed
        # If any route is invalid, we need to fix it.
        # The local search should maintain validity, but just in case:
        
        coords = instance["coords"]
        demands = np.array(instance["demands"], dtype=np.float64)
        ready = np.array(instance["ready"], dtype=np.float64)
        due = np.array(instance["due"], dtype=np.float64)
        service = np.array(instance["service"], dtype=np.float64)
        n_customers = instance["n_customers"]
        stations = instance["stations"]
        stations_set = set(stations)
        capacity = instance["capacity"]
        battery = instance["battery"]
        speed = instance["speed"]
        recharge_rate = instance["recharge_rate"]
        horizon = instance["horizon"]
        
        dist_matrix = _build_distance_matrix(coords)
        
        valid_routes = []
        served = set()
        
        for r in routes:
            _, feasible = _simulate_route(r, dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon)
            if feasible:
                valid_routes.append(r)
                for c in r:
                    if c not in stations_set:
                        served.add(c)
        
        # Check for unserved customers
        all_customers = set(range(1, n_customers + 1))
        unserved = all_customers - served
        
        # If unserved, add them as single routes (should be feasible)
        for c in unserved:
            # Try to insert into existing routes first
            inserted = False
            for ri, route in enumerate(valid_routes):
                for pos in range(len(route) + 1):
                    new_route = route[:pos] + [c] + route[pos:]
                    _, feasible = _simulate_route(new_route, dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon)
                    if feasible:
                        valid_routes[ri] = new_route
                        inserted = True
                        break
                if inserted:
                    break
            if not inserted:
                valid_routes.append([c])
        
        return valid_routes

    except Exception:
        # Fallback: single customer routes
        customers = list(range(1, instance["n_customers"] + 1))
        customers.sort(key=lambda v: instance["due"][v])
        return [[v] for v in customers]
# EVOLVE-BLOCK-END
