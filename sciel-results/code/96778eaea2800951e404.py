# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random

def _euclidean(coords):
    n = len(coords)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = math.hypot(coords[i][0]-coords[j][0], coords[i][1]-coords[j][1])
            d[i][j] = dist
            d[j][i] = dist
    return d

def _simulate_route(route, dist, demands, ready, due, service, stations,
                    capacity, battery_max, speed, recharge_rate, horizon, coords):
    """Simulate a route, return (feasible, total_distance, final_time)."""
    if not route:
        return True, 0.0, 0.0
    
    total_dist = 0.0
    time_t = 0.0
    battery = battery_max
    load = 0.0
    prev = 0  # depot
    
    for node in route:
        d = dist[prev][node]
        total_dist += d
        time_t += d / speed
        battery -= d
        
        if node in stations:
            # Charging station: refill battery
            energy = battery
            time_t += max(0, (battery_max - energy)) * recharge_rate
            battery = battery_max
        else:
            # Customer
            if battery < 0:
                return False, total_dist, time_t
            if load + demands[node] > capacity:
                return False, total_dist, time_t
            # Time window
            if time_t < ready[node]:
                time_t = ready[node]
            if time_t > due[node]:
                return False, total_dist, time_t
            time_t += service[node]
            load += demands[node]
        
        prev = node
    
    # Return to depot
    d = dist[prev][0]
    total_dist += d
    time_t += d / speed
    battery -= d
    if battery < 0:
        return False, total_dist, time_t
    if time_t > horizon:
        return False, total_dist, time_t
    
    return True, total_dist, time_t

def _route_distance(route, dist):
    """Compute total distance of a route including depot legs."""
    if not route:
        return 0.0
    total = dist[0][route[0]]
    for i in range(len(route)-1):
        total += dist[route[i]][route[i+1]]
    total += dist[route[-1]][0]
    return total

def _constructive_solve(coords, demands, ready, due, service, n_customers,
                        stations, capacity, battery_max, speed, recharge_rate, horizon):
    """Constructive heuristic: nearest neighbor with feasibility checks."""
    n_nodes = len(coords)
    dist = _euclidean(coords)
    
    customers = list(range(1, n_customers+1))
    served = set()
    routes = []
    
    # Sort customers by due time for initial ordering
    customers_sorted = sorted(customers, key=lambda c: due[c])
    
    # Try to build routes greedily
    remaining = list(customers_sorted)
    
    while remaining:
        # Start a new route with the customer having earliest due time
        start = remaining[0]
        route = [start]
        served.add(start)
        remaining.remove(start)
        
        # Try to add more customers
        while remaining:
            best_cand = None
            best_dist = float('inf')
            
            for c in remaining:
                # Check if we can add c to current route
                new_route = route + [c]
                feasible, total_d, t = _simulate_route(
                    new_route, dist, demands, ready, due, service,
                    stations, capacity, battery_max, speed, recharge_rate, horizon, coords)
                if feasible:
                    d = _route_distance(new_route, dist)
                    if d < best_dist:
                        best_dist = d
                        best_cand = c
            
            if best_cand is None:
                break
            
            route.append(best_cand)
            served.add(best_cand)
            remaining.remove(best_cand)
        
        routes.append(route)
    
    return routes

def _insert_customer(routes, customer, dist, demands, ready, due, service,
                     stations, capacity, battery_max, speed, recharge_rate, horizon, coords):
    """Try to insert a customer into an existing route at best position."""
    best_route_idx = -1
    best_pos = -1
    best_delta = float('inf')
    
    for ri, route in enumerate(routes):
        # Try all insertion positions
        for pos in range(len(route)+1):
            new_route = route[:pos] + [customer] + route[pos:]
            feasible, total_d, t = _simulate_route(
                new_route, dist, demands, ready, due, service,
                stations, capacity, battery_max, speed, recharge_rate, horizon, coords)
            if feasible:
                delta = _route_distance(new_route, dist) - _route_distance(route, dist)
                if delta < best_delta:
                    best_delta = delta
                    best_route_idx = ri
                    best_pos = pos
    
    if best_route_idx >= 0:
        route = routes[best_route_idx]
        routes[best_route_idx] = route[:best_pos] + [customer] + route[best_pos:]
        return True
    return False

def _local_search(routes, dist, demands, ready, due, service, stations,
                  capacity, battery_max, speed, recharge_rate, horizon, coords,
                  max_iters=200, time_limit=10.0):
    """Local search with 2-opt, swap, and relocate moves."""
    start_time = time.time()
    n_customers = max(demands) if demands else 0
    
    def total_distance(routes):
        return sum(_route_distance(r, dist) for r in routes)
    
    current_dist = total_distance(routes)
    no_improve = 0
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        improved = False
        
        # 2-opt within each route
        for ri in range(len(routes)):
            route = routes[ri]
            for i in range(len(route)-1):
                for j in range(i+1, len(route)):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    feasible, total_d, t = _simulate_route(
                        new_route, dist, demands, ready, due, service,
                        stations, capacity, battery_max, speed, recharge_rate, horizon, coords)
                    if feasible:
                        new_dist = total_distance(routes) - _route_distance(route, dist) + _route_distance(new_route, dist)
                        if new_dist < current_dist - 1e-9:
                            routes[ri] = new_route
                            current_dist = new_dist
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            no_improve = 0
            continue
        
        # Relocate: move a customer from one route to another
        improved = False
        for ri in range(len(routes)):
            for ci in range(len(routes[ri])):
                c = routes[ri][ci]
                # Try inserting into other routes
                for rj in range(len(routes)):
                    if rj == ri:
                        continue
                    route_j = routes[rj]
                    for pos in range(len(route_j)+1):
                        new_route_j = route_j[:pos] + [c] + route_j[pos:]
                        feasible, total_d, t = _simulate_route(
                            new_route_j, dist, demands, ready, due, service,
                            stations, capacity, battery_max, speed, recharge_rate, horizon, coords)
                        if feasible:
                            # Remove c from route i
                            route_i = routes[ri]
                            new_route_i = route_i[:ci] + route_i[ci+1:]
                            # Check route_i still feasible (it should be since we're removing)
                            if new_route_i:
                                feas_i, _, _ = _simulate_route(
                                    new_route_i, dist, demands, ready, due, service,
                                    stations, capacity, battery_max, speed, recharge_rate, horizon, coords)
                                if not feas_i:
                                    continue
                            else:
                                # Route becomes empty
                                pass
                            
                            old_dist = _route_distance(route_i, dist) + _route_distance(route_j, dist)
                            new_dist = _route_distance(new_route_i, dist) + _route_distance(new_route_j, dist)
                            if new_dist < current_dist - 1e-9:
                                routes[ri] = new_route_i
                                routes[rj] = new_route_j
                                current_dist = total_distance(routes)
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            no_improve = 0
            continue
        
        # Swap: swap two customers between routes
        improved = False
        for ri in range(len(routes)):
            for rj in range(ri+1, len(routes)):
                route_i = routes[ri]
                route_j = routes[rj]
                for ci in range(len(route_i)):
                    for cj in range(len(route_j)):
                        c1 = route_i[ci]
                        c2 = route_j[cj]
                        new_route_i = route_i[:ci] + [c2] + route_i[ci+1:]
                        new_route_j = route_j[:cj] + [c1] + route_j[cj+1:]
                        feas_i, _, _ = _simulate_route(
                            new_route_i, dist, demands, ready, due, service,
                            stations, capacity, battery_max, speed, recharge_rate, horizon, coords)
                        feas_j, _, _ = _simulate_route(
                            new_route_j, dist, demands, ready, due, service,
                            stations, capacity, battery_max, speed, recharge_rate, horizon, coords)
                        if feas_i and feas_j:
                            old_dist = _route_distance(route_i, dist) + _route_distance(route_j, dist)
                            new_dist = _route_distance(new_route_i, dist) + _route_distance(new_route_j, dist)
                            if new_dist < current_dist - 1e-9:
                                routes[ri] = new_route_i
                                routes[rj] = new_route_j
                                current_dist = total_distance(routes)
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            no_improve += 1
            if no_improve >= 3:
                break
    
    return routes

def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    # Precompute distance matrix
    dist = _euclidean(coords)
    
    # Phase 1: Constructive solution
    routes = _constructive_solve(
        coords, demands, ready, due, service, n_customers,
        stations, capacity, battery_max, speed, recharge_rate, horizon)
    
    # Verify all customers served
    all_served = set()
    for r in routes:
        for c in r:
            all_served.add(c)
    expected = set(range(1, n_customers+1))
    if all_served != expected:
        # Fallback: one customer per route
        routes = [[c] for c in range(1, n_customers+1)]
    
    # Phase 2: Local search
    if time.time() < deadline - 5.0:
        routes = _local_search(
            routes, dist, demands, ready, due, service, stations,
            capacity, battery_max, speed, recharge_rate, horizon, coords,
            max_iters=100, time_limit=30.0)
    
    # Phase 3: Another round of local search if time permits
    if time.time() < deadline - 3.0:
        routes = _local_search(
            routes, dist, demands, ready, due, service, stations,
            capacity, battery_max, speed, recharge_rate, horizon, coords,
            max_iters=50, time_limit=15.0)
    
    # Final verification
    for r in routes:
        feasible, _, _ = _simulate_route(
            r, dist, demands, ready, due, service,
            stations, capacity, battery_max, speed, recharge_rate, horizon, coords)
        if not feasible:
            # This shouldn't happen, but fallback
            routes = [[c] for c in range(1, n_customers+1)]
            break
    
    return routes
# EVOLVE-BLOCK-END
