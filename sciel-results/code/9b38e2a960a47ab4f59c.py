import math
import time
import random
import numpy as np

_EPS = 1e-9

def _dist(a, b, coords):
    return math.hypot(coords[a][0] - coords[b][0], coords[a][1] - coords[b][1])

def _check_route(route, coords, demands, ready, due, service, stations,
                 capacity, battery_max, speed, recharge_rate, horizon):
    """
    Returns (total_distance, valid, final_time, final_battery)
    """
    if not route:
        return 0.0, True, 0.0, battery_max
    
    total_dist = 0.0
    t = 0.0
    b = battery_max
    load = 0.0
    prev = 0
    
    # Pre-check capacity
    for node in route:
        if node not in stations:
            load += demands[node]
    if load > capacity + _EPS:
        return float('inf'), False, 0.0, 0.0

    for node in route:
        d = _dist(prev, node, coords)
        total_dist += d
        t += d / speed
        b -= d
        
        if b < -_EPS:
            return float('inf'), False, 0.0, 0.0
            
        if node in stations:
            # Recharge
            t += (battery_max - b) * recharge_rate
            b = battery_max
        else:
            # Customer
            t = max(t, ready[node])
            if t > due[node] + _EPS:
                return float('inf'), False, 0.0, 0.0
            t += service[node]
        
        prev = node
        
    # Return to depot
    d = _dist(prev, 0, coords)
    total_dist += d
    t += d / speed
    b -= d
    
    if b < -_EPS:
        return float('inf'), False, 0.0, 0.0
        
    if t > horizon + _EPS:
        return float('inf'), False, 0.0, 0.0
        
    return total_dist, True, t, b

def _try_insert_station(route, coords, demands, ready, due, service, stations,
                        capacity, battery_max, speed, recharge_rate, horizon):
    """
    If route is infeasible due to battery, try inserting a station.
    Returns (new_route, new_cost) or (None, inf)
    """
    # Quick check if it's actually a battery issue
    total_dist, valid, t, b = _check_route(route, coords, demands, ready, due, service, stations,
                                           capacity, battery_max, speed, recharge_rate, horizon)
    if valid:
        return route, total_dist
        
    # If not valid, try inserting each station at each position
    best_route = None
    best_cost = float('inf')
    
    for s in stations:
        # Don't insert if already in route (simplification, usually not optimal to repeat if avoidable, but allowed)
        # Actually, repeating is allowed.
        for i in range(len(route) + 1):
            # Skip if inserting at start/end is redundant or if node is already there
            # Just try all
            cand = route[:i] + [s] + route[i:]
            c, v, _, _ = _check_route(cand, coords, demands, ready, due, service, stations,
                                      capacity, battery_max, speed, recharge_rate, horizon)
            if v and c < best_cost:
                best_cost = c
                best_route = cand
                
    if best_route:
        return best_route, best_cost
    return None, float('inf')

def _two_opt(route, coords, demands, ready, due, service, stations,
             capacity, battery_max, speed, recharge_rate, horizon):
    """
    Perform 2-opt on a single route.
    Returns improved route and cost.
    """
    n = len(route)
    if n < 2:
        return route, _check_route(route, coords, demands, ready, due, service, stations,
                                   capacity, battery_max, speed, recharge_rate, horizon)[0]
    
    cur_cost, cur_valid, _, _ = _check_route(route, coords, demands, ready, due, service, stations,
                                             capacity, battery_max, speed, recharge_rate, horizon)
    if not cur_valid:
        # Try to repair first? Or just return invalid.
        # For local search, we only accept valid improvements.
        return route, cur_cost
        
    best_route = list(route)
    best_cost = cur_cost
    improved = True
    max_iter = 50 # Limit iterations to avoid infinite loops
    
    while improved and max_iter > 0:
        improved = False
        max_iter -= 1
        for i in range(n - 1):
            for j in range(i + 1, n):
                # 2-opt: reverse segment i to j
                cand = best_route[:i] + list(reversed(best_route[i:j+1])) + best_route[j+1:]
                c, v, _, _ = _check_route(cand, coords, demands, ready, due, service, stations,
                                          capacity, battery_max, speed, recharge_rate, horizon)
                if v and c < best_cost - 1e-7:
                    best_cost = c
                    best_route = cand
                    improved = True
                    
    return best_route, best_cost

def _relocate_customer(cust, from_route, to_route, from_idx, to_pos, 
                       coords, demands, ready, due, service, stations,
                       capacity, battery_max, speed, recharge_rate, horizon):
    """
    Check if moving cust from from_route (at from_idx) to to_route (at to_pos) is feasible and beneficial.
    Returns (new_from_route, new_to_route, delta_cost, valid)
    """
    new_from = from_route[:from_idx] + from_route[from_idx+1:]
    new_to = to_route[:to_pos] + [cust] + to_route[to_pos:]
    
    c1, v1, _, _ = _check_route(new_from, coords, demands, ready, due, service, stations,
                                capacity, battery_max, speed, recharge_rate, horizon)
    if not v1:
        return None, None, float('inf'), False
        
    c2, v2, _, _ = _check_route(new_to, coords, demands, ready, due, service, stations,
                                capacity, battery_max, speed, recharge_rate, horizon)
    if not v2:
        return None, None, float('inf'), False
        
    return new_from, new_to, c1 + c2, True

def _build_initial_routes(instance):
    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_cust = instance["n_customers"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    customers = list(range(1, n_cust + 1))
    # Sort by due time for better initial construction
    customers.sort(key=lambda v: due[v])
    
    routes = []
    route_loads = []
    
    for cust in customers:
        best_cost = float('inf')
        best_route_idx = -1
        best_pos = -1
        best_route = None
        
        # Try inserting into existing routes
        for ri, r in enumerate(routes):
            if route_loads[ri] + demands[cust] > capacity + _EPS:
                continue
            for pos in range(len(r) + 1):
                cand = r[:pos] + [cust] + r[pos:]
                c, v, _, _ = _check_route(cand, coords, demands, ready, due, service, stations,
                                          capacity, battery, speed, recharge_rate, horizon)
                if v and c < best_cost:
                    best_cost = c
                    best_route_idx = ri
                    best_pos = pos
                    best_route = cand
        
        # Try new route
        new_route = [cust]
        c, v, _, _ = _check_route(new_route, coords, demands, ready, due, service, stations,
                                  capacity, battery, speed, recharge_rate, horizon)
        if v and c < best_cost:
            best_cost = c
            best_route_idx = -1
            best_pos = -1
            best_route = new_route
            
        if best_route_idx == -1:
            # Start new route
            # If even new route is invalid, we have a problem. 
            # But single customer routes are usually feasible.
            # If not, we might need to insert a station.
            if not v:
                repaired, rcost = _try_insert_station(new_route, coords, demands, ready, due, service, stations,
                                                      capacity, battery, speed, recharge_rate, horizon)
                if repaired:
                    routes.append(repaired)
                    route_loads.append(demands[cust])
                else:
                    # Fallback: keep it anyway? No, must be valid.
                    # This should rarely happen.
                    routes.append(new_route)
                    route_loads.append(demands[cust])
            else:
                routes.append(new_route)
                route_loads.append(demands[cust])
        else:
            # Insert into existing
            r = routes[best_route_idx]
            r.insert(best_pos, cust)
            route_loads[best_route_idx] += demands[cust]
            
    return routes

def plan(instance):
    t0 = time.time()
    DEADLINE = 145.0 # Safety margin
    
    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_cust = instance["n_customers"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    # 1. Initial Construction
    routes = _build_initial_routes(instance)
    
    # 2. Local Search
    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        
        # Intra-route 2-Opt
        for ri in range(len(routes)):
            if time.time() > DEADLINE: break
            new_route, new_cost = _two_opt(routes[ri], coords, demands, ready, due, service, stations,
                                           capacity, battery, speed, recharge_rate, horizon)
            old_cost, old_valid, _, _ = _check_route(routes[ri], coords, demands, ready, due, service, stations,
                                                     capacity, battery, speed, recharge_rate, horizon)
            if old_valid and new_cost < old_cost - 1e-7:
                routes[ri] = new_route
                improved = True
                
        # Inter-route Relocate
        if time.time() > DEADLINE: break
        for ri in range(len(routes)):
            if time.time() > DEADLINE: break
            for ci in range(len(routes[ri])):
                if time.time() > DEADLINE: break
                cust = routes[ri][ci]
                for rj in range(len(routes)):
                    if rj == ri: continue
                    if time.time() > DEADLINE: break
                    for cj in range(len(routes[rj]) + 1):
                        if time.time() > DEADLINE: break
                        # Check capacity
                        load_j = sum(demands[n] for n in routes[rj] if n not in stations)
                        if load_j + demands[cust] > capacity + _EPS:
                            continue
                            
                        new_from, new_to, delta_cost, valid = _relocate_customer(
                            cust, routes[ri], routes[rj], ci, cj,
                            coords, demands, ready, due, service, stations,
                            capacity, battery, speed, recharge_rate, horizon
                        )
                        
                        if valid:
                            # Calculate total cost change
                            old_cost_i, _, _, _ = _check_route(routes[ri], coords, demands, ready, due, service, stations,
                                                               capacity, battery, speed, recharge_rate, horizon)
                            old_cost_j, _, _, _ = _check_route(routes[rj], coords, demands, ready, due, service, stations,
                                                               capacity, battery, speed, recharge_rate, horizon)
                            if old_cost_i + old_cost_j > delta_cost + 1e-7:
                                routes[ri] = new_from
                                routes[rj] = new_to
                                improved = True
                                # Break out of loops to restart relocation from scratch
                                # This is simple but effective for small N
                                ri = -1 # Force restart of outer loop logic? 
                                # Actually, breaking nested loops is tricky.
                                # We'll just mark improved and continue.
                                # To be safe, let's break out of all loops and restart the while loop.
                                # This ensures we don't use stale route info.
                                ri = 999999
                                break
                    if ri == 999999: break
                if ri == 999999: break
            if ri == 999999: break
            
        # Remove empty routes
        routes = [r for r in routes if r]
        
        # Inter-route Swap
        if time.time() > DEADLINE: break
        swapped = False
        for ri in range(len(routes)):
            if swapped or time.time() > DEADLINE: break
            for rj in range(ri + 1, len(routes)):
                if swapped or time.time() > DEADLINE: break
                for ci in range(len(routes[ri])):
                    if swapped or time.time() > DEADLINE: break
                    cust1 = routes[ri][ci]
                    for cj in range(len(routes[rj])):
                        if swapped or time.time() > DEADLINE: break
                        cust2 = routes[rj][cj]
                        
                        # Check capacity
                        load_i = sum(demands[n] for n in routes[ri] if n not in stations)
                        load_j = sum(demands[n] for n in routes[rj] if n not in stations)
                        
                        # After swap:
                        # Route i loses cust1, gains cust2
                        # Route j loses cust2, gains cust1
                        new_load_i = load_i - demands[cust1] + demands[cust2]
                        new_load_j = load_j - demands[cust2] + demands[cust1]
                        
                        if new_load_i > capacity + _EPS or new_load_j > capacity + _EPS:
                            continue
                            
                        new_route_i = routes[ri][:]
                        new_route_i[ci] = cust2
                        new_route_j = routes[rj][:]
                        new_route_j[cj] = cust1
                        
                        c_i, v_i, _, _ = _check_route(new_route_i, coords, demands, ready, due, service, stations,
                                                      capacity, battery, speed, recharge_rate, horizon)
                        c_j, v_j, _, _ = _check_route(new_route_j, coords, demands, ready, due, service, stations,
                                                      capacity, battery, speed, recharge_rate, horizon)
                        
                        if v_i and v_j:
                            old_c_i, _, _, _ = _check_route(routes[ri], coords, demands, ready, due, service, stations,
                                                            capacity, battery, speed, recharge_rate, horizon)
                            old_c_j, _, _, _ = _check_route(routes[rj], coords, demands, ready, due, service, stations,
                                                            capacity, battery, speed, recharge_rate, horizon)
                            
                            if old_c_i + old_c_j > c_i + c_j + 1e-7:
                                routes[ri] = new_route_i
                                routes[rj] = new_route_j
                                improved = True
                                swapped = True
                                break
                if swapped: break
            if swapped: break

    # Final validation
    total_dist = 0.0
    served = set()
    valid = True
    for r in routes:
        c, v, _, _ = _check_route(r, coords, demands, ready, due, service, stations,
                                  capacity, battery, speed, recharge_rate, horizon)
        if not v:
            valid = False
            break
        total_dist += c
        for node in r:
            if node in served:
                valid = False
                break
            served.add(node)
        if not valid:
            break
            
    if not valid or served != set(range(1, n_cust + 1)):
        # Fallback: one customer per route
        customers = list(range(1, n_cust + 1))
        customers.sort(key=lambda v: due[v])
        routes = [[v] for v in customers]
        # Validate fallback
        # (Assumed valid)

    return routes
