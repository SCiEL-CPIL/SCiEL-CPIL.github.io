"""Optimized EV routing with time windows and battery constraints."""
# EVOLVE-BLOCK-START
import numpy as np
import math
import time
import random

DEADLINE_SECONDS = 115.0


def _euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _build_distance_matrix(coords):
    n = len(coords)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = math.sqrt((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2)
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _simulate_route(route, instance, dist):
    """Simulate a route and return (total_distance, is_feasible, battery_min)."""
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])
    
    # Check capacity
    total_demand = sum(demands[v] for v in route if v not in stations)
    if total_demand > capacity:
        return float('inf'), False, 0.0
    
    battery = battery_max
    time_curr = 0.0
    total_dist = 0.0
    current = 0  # depot
    
    for v in route:
        d = dist[current, v]
        time_curr += d / speed
        battery -= d
        
        if battery < -1e-9:
            return float('inf'), False, battery
        
        if v in stations:
            # Charge: adds (battery - energy) * recharge_rate time, refills
            # energy = 0 at station (we arrive with some battery)
            # The rule: station adds (battery - energy) * recharge_rate time and refills
            # I interpret: time added = (battery_max - battery) * recharge_rate, then battery = battery_max
            charge_time = (battery_max - battery) * recharge_rate
            time_curr += charge_time
            battery = battery_max
        else:
            # Customer: time = max(time, ready), must be <= due, then service added
            time_curr = max(time_curr, ready[v])
            if time_curr > due[v] + 1e-9:
                return float('inf'), False, battery
            time_curr += service[v]
        
        total_dist += d
        current = v
    
    # Return to depot
    d = dist[current, 0]
    time_curr += d / speed
    battery -= d
    total_dist += d
    
    if battery < -1e-9:
        return float('inf'), False, battery
    
    if time_curr > horizon + 1e-9:
        return float('inf'), False, battery
    
    return total_dist, True, battery


def _try_insert_station(route, station, instance, dist):
    """Try to insert a charging station at the best position in route.
    Returns (new_route, improvement) or (None, 0) if no improvement."""
    best_pos = None
    best_cost = float('inf')
    best_new_route = None
    
    n = len(route)
    # Try inserting at each position (including beginning and end)
    for pos in range(n + 1):
        new_route = route[:pos] + [station] + route[pos:]
        dist_val, feas, _ = _simulate_route(new_route, instance, dist)
        if feas and dist_val < best_cost:
            best_cost = dist_val
            best_pos = pos
            best_new_route = new_route
    
    if best_new_route is not None:
        return best_new_route, best_cost
    return None, float('inf')


def _construct_initial_routes(instance, dist):
    """Construct initial feasible routes using nearest neighbor with time/battery awareness."""
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])
    
    customers = list(range(1, instance["n_customers"] + 1))
    # Sort by due time for initial construction
    customers.sort(key=lambda v: due[v])
    
    routes = []
    unassigned = set(customers)
    
    while unassigned:
        route = []
        current = 0
        battery = battery_max
        time_curr = 0.0
        load = 0
        
        while unassigned:
            best_customer = None
            best_cost = float('inf')
            
            for c in unassigned:
                if load + demands[c] > capacity:
                    continue
                
                d = dist[current, c]
                new_time = time_curr + d / speed
                new_battery = battery - d
                
                if new_battery < -1e-9:
                    continue
                
                if c in stations:
                    continue
                
                # Check if we need to charge before visiting c
                # For simplicity, just check if battery is sufficient
                new_time = max(new_time, ready[c])
                if new_time > due[c] + 1e-9:
                    continue
                new_time += service[c]
                
                # Check return to depot feasibility
                d_return = dist[c, 0]
                final_time = new_time + d_return / speed
                final_battery = new_battery - d_return
                if final_battery < -1e-9 or final_time > horizon + 1e-9:
                    continue
                
                cost = d
                if cost < best_cost:
                    best_cost = cost
                    best_customer = c
            
            if best_customer is None:
                # Try inserting a station to enable more customers
                station_inserted = False
                for s in stations:
                    if s in route:
                        continue
                    new_route, _ = _try_insert_station(route, s, instance, dist)
                    if new_route is not None:
                        # Re-simulate to get new state
                        # For simplicity, just break and try again
                        station_inserted = True
                        break
                
                if not station_inserted:
                    break
                
                # Re-evaluate with potential station
                # This is complex; for now, just break
                break
            
            # Visit best_customer
            d = dist[current, best_customer]
            time_curr += d / speed
            battery -= d
            load += demands[best_customer]
            
            if best_customer in stations:
                charge_time = (battery_max - battery) * recharge_rate
                time_curr += charge_time
                battery = battery_max
            else:
                time_curr = max(time_curr, ready[best_customer])
                time_curr += service[best_customer]
            
            route.append(best_customer)
            unassigned.remove(best_customer)
            current = best_customer
        
        if route:
            # Try to add stations if needed for feasibility
            dist_val, feas, _ = _simulate_route(route, instance, dist)
            if not feas:
                # Try adding stations
                improved = False
                for s in stations:
                    if s in route:
                        continue
                    new_route, new_dist = _try_insert_station(route, s, instance, dist)
                    if new_route is not None:
                        nd, nf, _ = _simulate_route(new_route, instance, dist)
                        if nf and nd < dist_val:
                            route = new_route
                            improved = True
                            break
                if not improved:
                    # Route is infeasible, skip it (shouldn't happen with good construction)
                    continue
            
            routes.append(route)
    
    return routes


def _two_opt_route(route, instance, dist):
    """Apply 2-opt to a single route, preserving feasibility."""
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])
    
    n = len(route)
    if n < 2:
        return route
    
    current_dist, current_feas, _ = _simulate_route(route, instance, dist)
    if not current_feas:
        return route
    
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                nd, nf, _ = _simulate_route(new_route, instance, dist)
                if nf and nd < current_dist - 1e-9:
                    route = new_route
                    current_dist = nd
                    improved = True
    
    return route


def _or_opt_route(route, instance, dist):
    """Apply or-opt (move 1 or 2 consecutive nodes) to a single route."""
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])
    
    n = len(route)
    if n < 2:
        return route
    
    current_dist, current_feas, _ = _simulate_route(route, instance, dist)
    if not current_feas:
        return route
    
    improved = True
    while improved:
        improved = False
        for seg_len in [1, 2]:
            for i in range(n - seg_len + 1):
                segment = route[i:i+seg_len]
                rest = route[:i] + route[i+seg_len:]
                for j in range(len(rest) + 1):
                    if i <= j < i + seg_len:
                        continue
                    new_route = rest[:j] + segment + rest[j:]
                    if new_route == route:
                        continue
                    nd, nf, _ = _simulate_route(new_route, instance, dist)
                    if nf and nd < current_dist - 1e-9:
                        route = new_route
                        current_dist = nd
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    return route


def _local_search(routes, instance, dist, deadline):
    """Apply local search to all routes."""
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])
    
    start_time = time.time()
    improved = True
    while improved:
        if time.time() - start_time > deadline:
            break
        improved = False
        
        # 2-opt on each route
        for i in range(len(routes)):
            if time.time() - start_time > deadline:
                break
            new_route = _two_opt_route(routes[i], instance, dist)
            if new_route != routes[i]:
                routes[i] = new_route
                improved = True
        
        # or-opt on each route
        for i in range(len(routes)):
            if time.time() - start_time > deadline:
                break
            new_route = _or_opt_route(routes[i], instance, dist)
            if new_route != routes[i]:
                routes[i] = new_route
                improved = True
    
    return routes


def _total_distance(routes, instance, dist):
    """Compute total distance of all routes."""
    total = 0.0
    for route in routes:
        d, feas, _ = _simulate_route(route, instance, dist)
        if feas:
            total += d
        else:
            return float('inf')
    return total


def _validate_solution(routes, instance, dist):
    """Validate that all customers are served exactly once and all routes are feasible."""
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])
    
    n_customers = instance["n_customers"]
    all_customers = set(range(1, n_customers + 1))
    served = set()
    
    for route in routes:
        for v in route:
            if v in all_customers:
                if v in served:
                    return False
                served.add(v)
            elif v not in stations:
                # Invalid node
                return False
        
        # Check route feasibility
        d, feas, _ = _simulate_route(route, instance, dist)
        if not feas:
            return False
    
    if served != all_customers:
        return False
    
    return True


def plan(instance):
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = instance["stations"]
    n_customers = instance["n_customers"]
    
    # Build distance matrix
    dist = _build_distance_matrix(coords)
    
    # Construct initial routes
    routes = _construct_initial_routes(instance, dist)
    
    # Validate initial solution
    if not _validate_solution(routes, instance, dist):
        # Fallback: one customer per route
        customers = list(range(1, n_customers + 1))
        customers.sort(key=lambda v: due[v])
        routes = [[v] for v in customers]
    
    # Local search
    routes = _local_search(routes, instance, dist, DEADLINE_SECONDS - 10)
    
    # Final validation
    if not _validate_solution(routes, instance, dist):
        # Fallback to one per route
        customers = list(range(1, n_customers + 1))
        customers.sort(key=lambda v: due[v])
        routes = [[v] for v in customers]
    
    return routes
# EVOLVE-BLOCK-END
