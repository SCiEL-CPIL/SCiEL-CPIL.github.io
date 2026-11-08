"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import sqrt


def _build_distance_matrix(coords):
    n = len(coords)
    dm = np.zeros((n, n))
    for i in range(n):
        dx = coords[i, 0] - coords[i+1:, 0]
        dy = coords[i, 1] - coords[i+1:, 1]
        dm[i, i+1:] = np.sqrt(dx * dx + dy * dy)
    dm = dm + dm.T
    return dm


def _simulate_route(route, dm, demands, ready, due, service, stations, capacity, battery, speed, recharge_rate, horizon):
    """
    Simulate a route (list of customer/station node ids, depot implicit at ends).
    Returns (is_valid, total_distance)
    """
    if not route:
        return True, 0.0
    
    cur_node = 0
    cur_time = 0.0
    cur_bat = battery
    total_dist = 0.0
    load = 0.0
    
    for node in route:
        d = dm[cur_node, node]
        cur_time += d / speed
        cur_bat -= d
        total_dist += d
        
        if cur_bat < -1e-9:
            return False, total_dist
        
        if node in stations:
            # At station: recharge
            energy_needed = battery - cur_bat
            cur_time += energy_needed * recharge_rate
            cur_bat = battery
        else:
            # At customer: check time window, add service
            if cur_time < ready[node]:
                cur_time = ready[node]
            if cur_time > due[node] + 1e-9:
                return False, total_dist
            cur_time += service[node]
            load += demands[node]
            if load > capacity + 1e-9:
                return False, total_dist
        
        cur_node = node
    
    # Return to depot
    d = dm[cur_node, 0]
    cur_time += d / speed
    cur_bat -= d
    total_dist += d
    
    if cur_bat < -1e-9:
        return False, total_dist
    if cur_time > horizon + 1e-9:
        return False, total_dist
    
    return True, total_dist


def _two_opt_improve(route, dm, demands, ready, due, service, stations, capacity, battery, speed, recharge_rate, horizon, deadline):
    """
    Apply 2-opt local search to a single route, preserving feasibility.
    Stations are treated as fixed anchors; we only reorder customers between stations.
    Returns improved route.
    """
    if len(route) < 2:
        return route
    
    # Identify customer-only segments (between depot/stations)
    # Build segments: each segment is a list of customers, separated by stations
    segments = []
    current_seg = []
    for node in route:
        if node in stations:
            if current_seg:
                segments.append(current_seg)
                current_seg = []
            segments.append([node])  # station as a segment marker
        else:
            current_seg.append(node)
    if current_seg:
        segments.append(current_seg)
    
    # For simplicity, do 2-opt on the full route but check feasibility
    # We'll work with the full route and try reversing subsegments
    best_route = list(route)
    best_dist, _ = _simulate_route(best_route, dm, demands, ready, due, service, stations, capacity, battery, speed, recharge_rate, horizon)
    
    n = len(best_route)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                # Try reversing segment [i..j]
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                valid, dist = _simulate_route(new_route, dm, demands, ready, due, service, stations, capacity, battery, speed, recharge_rate, horizon)
                if valid and dist < best_dist - 1e-9:
                    best_route = new_route
                    best_dist = dist
                    improved = True
        if time.time() > deadline:
            break
    
    return best_route


def _solve_instance(instance, deadline):
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    # Precompute distance matrix
    dm = _build_distance_matrix(coords)
    
    customers = list(range(1, n_customers + 1))
    customers.sort(key=lambda v: due[v])
    
    routes = []
    unserved = set(customers)
    
    # Greedy nearest-neighbor insertion with battery/time-window simulation
    while unserved:
        # Start a new route
        current_route = []
        cur_node = 0
        cur_time = 0.0
        cur_bat = battery
        cur_load = 0.0
        
        while unserved:
            best_cust = None
            best_dist = float('inf')
            
            for cust in unserved:
                d = dm[cur_node, cust]
                if d < best_dist:
                    # Check if we can serve this customer
                    test_time = cur_time + d / speed
                    test_bat = cur_bat - d
                    if test_bat < -1e-9:
                        continue
                    if test_time < ready[cust]:
                        test_time = ready[cust]
                    if test_time > due[cust] + 1e-9:
                        continue
                    test_time += service[cust]
                    if cur_load + demands[cust] > capacity + 1e-9:
                        continue
                    best_cust = cust
                    best_dist = d
            
            if best_cust is None:
                break
            
            # Move to best customer
            d = dm[cur_node, best_cust]
            cur_time += d / speed
            cur_bat -= d
            if cur_time < ready[best_cust]:
                cur_time = ready[best_cust]
            cur_time += service[best_cust]
            cur_load += demands[best_cust]
            cur_node = best_cust
            current_route.append(best_cust)
            unserved.discard(best_cust)
        
        # Check if we can return to depot
        d = dm[cur_node, 0]
        test_time = cur_time + d / speed
        test_bat = cur_bat - d
        if test_bat < -1e-9 or test_time > horizon + 1e-9:
            # Need to insert a station before returning
            # Try inserting the nearest station
            best_station = None
            best_sdist = float('inf')
            for st in stations:
                d_st = dm[cur_node, st]
                if d_st < best_sdist:
                    # Check if we can reach the station
                    if cur_bat - d_st >= -1e-9:
                        best_station = st
                        best_sdist = d_st
            
            if best_station is not None:
                # Go to station
                d = dm[cur_node, best_station]
                cur_time += d / speed
                cur_bat -= d
                # Recharge
                energy_needed = battery - cur_bat
                cur_time += energy_needed * recharge_rate
                cur_bat = battery
                cur_node = best_station
                current_route.append(best_station)
                
                # Now try to return to depot
                d = dm[cur_node, 0]
                test_time = cur_time + d / speed
                test_bat = cur_bat - d
                if test_bat < -1e-9 or test_time > horizon + 1e-9:
                    # Still can't return; this route is invalid
                    # Put back the last customer
                    last = current_route[-1]
                    if last not in stations:
                        unserved.add(last)
                        current_route.pop()
                    elif len(current_route) > 1:
                        # Last was a station, try removing it
                        current_route.pop()
                        cur_node = current_route[-1] if current_route else 0
                    continue
            else:
                # Can't reach any station; put back the last customer
                last = current_route[-1]
                if last not in stations:
                    unserved.add(last)
                    current_route.pop()
                continue
        
        routes.append(current_route)
    
    # Verify all routes are valid
    for route in routes:
        valid, _ = _simulate_route(route, dm, demands, ready, due, service, stations, capacity, battery, speed, recharge_rate, horizon)
        if not valid:
            # Fallback: one customer per route
            customers = list(range(1, n_customers + 1))
            customers.sort(key=lambda v: due[v])
            return [[v] for v in customers]
    
    # Apply 2-opt improvement to each route
    for idx, route in enumerate(routes):
        if time.time() > deadline:
            break
        if len(route) >= 2:
            improved = _two_opt_improve(route, dm, demands, ready, due, service, stations, capacity, battery, speed, recharge_rate, horizon, deadline)
            routes[idx] = improved
    
    # Final verification
    for route in routes:
        valid, _ = _simulate_route(route, dm, demands, ready, due, service, stations, capacity, battery, speed, recharge_rate, horizon)
        if not valid:
            # Fallback: one customer per route
            customers = list(range(1, n_customers + 1))
            customers.sort(key=lambda v: due[v])
            return [[v] for v in customers]
    
    return routes


def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s
    
    try:
        routes = _solve_instance(instance, deadline)
    except Exception:
        customers = list(range(1, instance["n_customers"] + 1))
        customers.sort(key=lambda v: instance["due"][v])
        return [[v] for v in customers]
    
    return routes
# EVOLVE-BLOCK-END
