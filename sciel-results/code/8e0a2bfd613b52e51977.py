"""EV routing with time windows – optimized construction + local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random
from itertools import combinations


def _build_distance_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
            D[i, j] = d
            D[j, i] = d
    return D


def _simulate(route, D, instance):
    """
    Simulate a single route. Returns (total_distance, valid, final_time, final_battery).
    """
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    n_customers = instance["n_customers"]

    depot = 0
    time_t = 0.0
    battery = battery_max
    load = 0
    total_dist = 0.0
    prev = depot

    for node in route:
        d = D[prev][node]
        time_t += d / speed
        battery -= d
        total_dist += d

        if battery < -1e-9:
            return total_dist, False, time_t, battery

        if node in stations:
            energy = battery_max - battery
            time_t += energy * recharge_rate
            battery = battery_max
        else:
            if node > n_customers:
                return total_dist, False, time_t, battery
            if time_t < ready[node]:
                time_t = ready[node]
            if time_t > due[node] + 1e-9:
                return total_dist, False, time_t, battery
            time_t += service[node]
            load += demands[node]
            if load > capacity + 1e-9:
                return total_dist, False, time_t, battery

        prev = node

    d = D[prev][depot]
    time_t += d / speed
    battery -= d
    total_dist += d

    if battery < -1e-9:
        return total_dist, False, time_t, battery
    if time_t > horizon + 1e-9:
        return total_dist, False, time_t, battery

    return total_dist, True, time_t, battery


def _total_distance(routes, D):
    depot = 0
    total = 0.0
    for route in routes:
        prev = depot
        for node in route:
            total += D[prev][node]
            prev = node
        total += D[prev][depot]
    return total


def _is_valid_solution(routes, D, instance):
    n_customers = instance["n_customers"]
    served = set()
    for route in routes:
        for node in route:
            if node > n_customers:
                return False
            if node in served:
                return False
            served.add(node)
    if len(served) != n_customers:
        return False

    for route in routes:
        _, valid, _, _ = _simulate(route, D, instance)
        if not valid:
            return False
    return True


def _clarke_wright_savings(instance, D):
    """
    Clarke-Wright savings heuristic for initial route construction.
    """
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    n_customers = instance["n_customers"]

    depot = 0
    # Start with each customer as its own route
    routes = [[i] for i in range(1, n_customers + 1)]
    
    # Compute savings
    savings = []
    for i in range(1, n_customers + 1):
        for j in range(i + 1, n_customers + 1):
            s = D[depot][i] + D[depot][j] - D[i][j]
            savings.append((s, i, j))
    savings.sort(reverse=True)

    # Map from customer to route index
    cust_to_route = {i: i - 1 for i in range(1, n_customers + 1)}
    
    for s, i, j in savings:
        if s <= 0:
            break
        ri = cust_to_route.get(i)
        rj = cust_to_route.get(j)
        if ri is None or rj is None or ri == rj:
            continue
        
        route_i = routes[ri]
        route_j = routes[rj]
        
        # Try to merge: i at end of route_i, j at end of route_j
        # Or i at start, j at end, etc.
        merged = None
        best_merge = None
        
        # Option 1: route_i + route_j (i at end of route_i, j at start of route_j)
        if route_i[-1] == i and route_j[0] == j:
            candidate = route_i + route_j
            merged = candidate
        # Option 2: route_i + reversed(route_j)
        elif route_i[-1] == i and route_j[-1] == j:
            candidate = route_i + route_j[::-1]
            merged = candidate
        # Option 3: reversed(route_i) + route_j
        elif route_i[0] == i and route_j[0] == j:
            candidate = route_i[::-1] + route_j
            merged = candidate
        # Option 4: reversed(route_i) + reversed(route_j)
        elif route_i[0] == i and route_j[-1] == j:
            candidate = route_i[::-1] + route_j[::-1]
            merged = candidate
        
        if merged is None:
            continue
        
        # Check if merged route is feasible
        total_demand = sum(demands[c] for c in merged)
        if total_demand > capacity + 1e-9:
            continue
        
        _, valid, _, _ = _simulate(merged, D, instance)
        if not valid:
            continue
        
        # Perform merge
        routes[ri] = merged
        routes[rj] = []
        for c in merged:
            cust_to_route[c] = ri
    
    # Remove empty routes
    routes = [r for r in routes if r]
    return routes


def _insert_station_if_needed(route, D, instance):
    """
    Insert charging stations into route where needed to maintain battery >= 0.
    Returns the modified route.
    """
    coords = instance["coords"]
    stations = instance["stations"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    n_customers = instance["n_customers"]

    depot = 0
    # Check if route is feasible without stations
    _, valid, _, _ = _simulate(route, D, instance)
    if valid:
        return route

    # Simulate and track battery
    time_t = 0.0
    battery = battery_max
    prev = depot
    new_route = []
    
    for node in route:
        d = D[prev][node]
        battery -= d
        time_t += d / speed
        
        if battery < -1e-9:
            # Need to insert a station before this node
            # Find nearest reachable station from prev
            best_station = None
            best_dist = float('inf')
            for st in stations:
                d_st = D[prev][st]
                if battery + d_st >= -1e-9:  # Can reach station
                    if d_st < best_dist:
                        best_dist = d_st
                        best_station = st
            
            if best_station is not None:
                new_route.append(best_station)
                # Update state
                battery = battery_max
                time_t += best_dist / speed
                energy = battery_max - battery
                time_t += energy * recharge_rate
                battery = battery_max
                prev = best_station
                # Continue to node
                d2 = D[prev][node]
                battery -= d2
                time_t += d2 / speed
            else:
                # Cannot fix with single station; try two
                # For simplicity, just return original
                return route
        
        new_route.append(node)
        if node not in set(stations):
            if time_t < ready[node]:
                time_t = ready[node]
            time_t += service[node]
        else:
            energy = battery_max - battery
            time_t += energy * recharge_rate
            battery = battery_max
        prev = node

    # Check return to depot
    d = D[prev][depot]
    battery -= d
    time_t += d / speed

    if battery < -1e-9:
        # Need station before depot
        best_station = None
        best_dist = float('inf')
        for st in stations:
            d_st = D[prev][st]
            if battery + d_st >= -1e-9:
                if d_st < best_dist:
                    best_dist = d_st
                    best_station = st
        if best_station is not None:
            new_route.append(best_station)

    return new_route


def _two_opt(route, D, instance, max_iters=50):
    """
    2-opt local search within a single route.
    """
    n = len(route)
    if n < 2:
        return route
    
    best_route = list(route)
    best_dist = _total_distance([best_route], D)
    
    for it in range(max_iters):
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                if j == i + 1:
                    continue
                # Reverse segment i..j
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                _, valid, _, _ = _simulate(new_route, D, instance)
                if not valid:
                    continue
                d = _total_distance([new_route], D)
                if d < best_dist - 1e-9:
                    best_route = new_route
                    best_dist = d
                    improved = True
        if not improved:
            break
    
    return best_route


def _or_opt(route, D, instance, max_iters=30):
    """
    Or-opt: relocate 1-3 consecutive customers.
    """
    best_route = list(route)
    best_dist = _total_distance([best_route], D)
    n = len(best_route)
    
    for it in range(max_iters):
        improved = False
        for start in range(n):
            for length in range(1, 4):
                if start + length > n:
                    break
                segment = best_route[start:start+length]
                remaining = best_route[:start] + best_route[start+length:]
                if not remaining:
                    # Put segment back
                    continue
                # Try inserting segment at each position in remaining
                for pos in range(len(remaining) + 1):
                    new_route = remaining[:pos] + segment + remaining[pos:]
                    _, valid, _, _ = _simulate(new_route, D, instance)
                    if not valid:
                        continue
                    d = _total_distance([new_route], D)
                    if d < best_dist - 1e-9:
                        best_route = new_route
                        best_dist = d
                        improved = True
                        break
        if not improved:
            break
    
    return best_route


def _cross_route_swap(routes, D, instance, max_iters=30):
    """
    Swap customers between two routes.
    """
    best_routes = [list(r) for r in routes]
    best_dist = _total_distance(best_routes, D)
    n_customers = instance["n_customers"]
    
    for it in range(max_iters):
        improved = False
        for ri in range(len(best_routes)):
            for rj in range(ri + 1, len(best_routes)):
                for i in range(len(best_routes[ri])):
                    for j in range(len(best_routes[rj])):
                        a = best_routes[ri][i]
                        b = best_routes[rj][j]
                        if a == b:
                            continue
                        new_ri = list(best_routes[ri])
                        new_rj = list(best_routes[rj])
                        new_ri[i] = b
                        new_rj[j] = a
                        
                        # Check validity of both routes
                        _, v1, _, _ = _simulate(new_ri, D, instance)
                        if not v1:
                            continue
                        _, v2, _, _ = _simulate(new_rj, D, instance)
                        if not v2:
                            continue
                        
                        # Check all customers served exactly once
                        all_nodes = []
                        for r in best_routes:
                            if r is best_routes[ri]:
                                all_nodes.extend(new_ri)
                            elif r is best_routes[rj]:
                                all_nodes.extend(new_rj)
                            else:
                                all_nodes.extend(r)
                        if len(all_nodes) != n_customers or len(set(all_nodes)) != n_customers:
                            continue
                        if any(n > n_customers for n in all_nodes):
                            continue
                        
                        d = _total_distance(
                            [new_ri if idx == ri else (new_rj if idx == rj else best_routes[idx]) 
                             for idx in range(len(best_routes))], D)
                        if d < best_dist - 1e-9:
                            best_routes[ri] = new_ri
                            best_routes[rj] = new_rj
                            best_dist = d
                            improved = True
        if not improved:
            break
    
    return best_routes


def _relocate_customer(routes, D, instance, max_iters=30):
    """
    Relocate a customer from one route to another.
    """
    best_routes = [list(r) for r in routes]
    best_dist = _total_distance(best_routes, D)
    n_customers = instance["n_customers"]
    
    for it in range(max_iters):
        improved = False
        for ri in range(len(best_routes)):
            for i in range(len(best_routes[ri])):
                cust = best_routes[ri][i]
                # Remove from route ri
                src_route = list(best_routes[ri])
                src_route.pop(i)
                
                for rj in range(len(best_routes)):
                    if rj == ri:
                        continue
                    dst_route = best_routes[rj]
                    # Try inserting at each position
                    for pos in range(len(dst_route) + 1):
                        new_dst = dst_route[:pos] + [cust] + dst_route[pos:]
                        
                        # Check validity
                        if src_route:
                            _, v1, _, _ = _simulate(src_route, D, instance)
                            if not v1:
                                continue
                        _, v2, _, _ = _simulate(new_dst, D, instance)
                        if not v2:
                            continue
                        
                        # Check all customers served exactly once
                        all_nodes = []
                        for r in best_routes:
                            if r is best_routes[ri]:
                                all_nodes.extend(src_route)
                            elif r is best_routes[rj]:
                                all_nodes.extend(new_dst)
                            else:
                                all_nodes.extend(r)
                        if len(all_nodes) != n_customers or len(set(all_nodes)) != n_customers:
                            continue
                        if any(n > n_customers for n in all_nodes):
                            continue
                        
                        d = _total_distance(
                            [src_route if idx == ri else (new_dst if idx == rj else best_routes[idx])
                             for idx in range(len(best_routes))], D)
                        if d < best_dist - 1e-9:
                            best_routes[ri] = src_route
                            best_routes[rj] = new_dst
                            best_dist = d
                            improved = True
                            break
                    if improved:
                        break
        if not improved:
            break
    
    # Remove empty routes
    best_routes = [r for r in best_routes if r]
    return best_routes


def plan(instance):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n_customers = instance["n_customers"]
    coords = instance["coords"]
    D = _build_distance_matrix(coords)
    
    # Generate initial solution using Clarke-Wright
    routes = _clarke_wright_savings(instance, D)
    
    # Insert stations if needed
    new_routes = []
    for r in routes:
        new_routes.append(_insert_station_if_needed(r, D, instance))
    
    # Check validity
    if not _is_valid_solution(new_routes, D, instance):
        # Fallback: one customer per route
        customers = list(range(1, n_customers + 1))
        customers.sort(key=lambda v: instance["due"][v])
        new_routes = [[v] for v in customers]
        new_routes = [_insert_station_if_needed(r, D, instance) for r in new_routes]
    
    best_routes = new_routes
    best_dist = _total_distance(best_routes, D)
    
    # Local search iterations
    while time.time() < deadline - 5:
        improved = False
        
        # 2-opt within each route
        for ri in range(len(best_routes)):
            if time.time() > deadline - 5:
                break
            new_route = _two_opt(best_routes[ri], D, instance, max_iters=20)
            if new_route != best_routes[ri]:
                _, valid, _, _ = _simulate(new_route, D, instance)
                if valid:
                    candidate = [new_route if idx == ri else best_routes[idx] for idx in range(len(best_routes))]
                    d = _total_distance(candidate, D)
                    if d < best_dist - 1e-9:
                        best_routes = candidate
                        best_dist = d
                        improved = True
        
        # Or-opt within each route
        if time.time() > deadline - 5:
            break
        for ri in range(len(best_routes)):
            if time.time() > deadline - 5:
                break
            new_route = _or_opt(best_routes[ri], D, instance, max_iters=10)
            if new_route != best_routes[ri]:
                _, valid, _, _ = _simulate(new_route, D, instance)
                if valid:
                    candidate = [new_route if idx == ri else best_routes[idx] for idx in range(len(best_routes))]
                    d = _total_distance(candidate, D)
                    if d < best_dist - 1e-9:
                        best_routes = candidate
                        best_dist = d
                        improved = True
        
        # Cross-route swap
        if time.time() > deadline - 5:
            break
        new_routes = _cross_route_swap(best_routes, D, instance, max_iters=10)
        if _is_valid_solution(new_routes, D, instance):
            d = _total_distance(new_routes, D)
            if d < best_dist - 1e-9:
                best_routes = new_routes
                best_dist = d
                improved = True
        
        # Relocate customer
        if time.time() > deadline - 5:
            break
        new_routes = _relocate_customer(best_routes, D, instance, max_iters=10)
        if _is_valid_solution(new_routes, D, instance):
            d = _total_distance(new_routes, D)
            if d < best_dist - 1e-9:
                best_routes = new_routes
                best_dist = d
                improved = True
        
        if not improved:
            break
    
    return best_routes
# EVOLVE-BLOCK-END
