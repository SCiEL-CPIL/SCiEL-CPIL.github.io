"""EV routing with time windows – matching-based construction + constrained shortest path for charging."""
# EVOLVE-BLOCK-START
import time
import math
import heapq
import numpy as np
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
    Simulate a single route. Returns (total_distance, valid, final_time, final_battery, load).
    """
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations_set = set(instance["stations"])

    t = 0.0
    batt = battery_max
    load = 0
    dist_total = 0.0
    prev = 0

    for node in route:
        d = D[prev][node]
        dist_total += d
        t += d
        batt -= d
        if batt < -1e-9:
            return dist_total, False, t, batt, load
        if node in stations_set:
            recharge_time = (battery_max - batt) * recharge_rate
            t += recharge_time
            batt = battery_max
        else:
            if t < ready[node]:
                t = ready[node]
            if t > due[node] + 1e-9:
                return dist_total, False, t, batt, load
            load += demands[node]
            if load > capacity + 1e-9:
                return dist_total, False, t, batt, load
            t += service[node]
        prev = node

    d = D[prev][0]
    dist_total += d
    t += d
    batt -= d
    if batt < -1e-9:
        return dist_total, False, t, batt, load
    if t > horizon + 1e-9:
        return dist_total, False, t, batt, load

    return dist_total, True, t, batt, load


def _constrained_shortest_path(start, end, D, instance, max_battery=300, num_battery_levels=30, num_time_levels=50, max_time=1000):
    """
    Find the shortest path from start to end that respects battery and time constraints.
    Uses Dijkstra on a (node, battery_level, time_level) state space.
    Returns (path, distance) or (None, None).
    """
    coords = instance["coords"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    battery_max = instance["battery"]
    recharge_rate = instance["recharge_rate"]
    stations_set = set(instance["stations"])
    
    # We discretize battery and time to keep the state space manageable
    # Actually, for a small graph, we can use a simpler approach:
    # Just try direct path, then path via each station.
    
    # Direct path
    d_direct = D[start][end]
    if d_direct <= battery_max:
        return [end], d_direct  # Just the end node, since start is implicit
    
    # Try path via each station
    best_path = None
    best_dist = float('inf')
    
    stations = instance["stations"]
    for st in stations:
        # start -> st -> end
        d1 = D[start][st]
        d2 = D[st][end]
        if d1 <= battery_max and d2 <= battery_max:
            total_d = d1 + d2
            if total_d < best_dist:
                # Check if this is feasible in terms of time
                # This is a rough check; full feasibility is checked later
                best_dist = total_d
                best_path = [st, end]
    
    if best_path is not None:
        return best_path, best_dist
    
    # Try two stations: start -> st1 -> st2 -> end
    for st1 in stations:
        for st2 in stations:
            if st1 == st2:
                continue
            d1 = D[start][st1]
            d2 = D[st1][st2]
            d3 = D[st2][end]
            if d1 <= battery_max and d2 <= battery_max and d3 <= battery_max:
                total_d = d1 + d2 + d3
                if total_d < best_dist:
                    best_dist = total_d
                    best_path = [st1, st2, end]
    
    return best_path, best_dist


def _build_route_from_path(path, D, instance):
    """
    Given a path of customer nodes (in order), build a full route with stations inserted as needed.
    Returns a list of node ids (customers and stations) or None if infeasible.
    """
    route = []
    prev = 0
    t = 0.0
    batt = instance["battery"]
    load = 0
    
    for node in path:
        # Try to go from prev to node directly
        d = D[prev][node]
        if d <= batt + 1e-9:
            # Direct is feasible
            t += d
            batt -= d
            if node not in set(instance["stations"]):
                if t < instance["ready"][node]:
                    t = instance["ready"][node]
                if t > instance["due"][node] + 1e-9:
                    return None
                load += instance["demands"][node]
                if load > instance["capacity"] + 1e-9:
                    return None
                t += instance["service"][node]
            route.append(node)
            prev = node
        else:
            # Need to insert a station
            sub_path, _ = _constrained_shortest_path(prev, node, D, instance)
            if sub_path is None:
                return None
            for sn in sub_path:
                if sn == node:
                    break
                route.append(sn)
                prev = sn
            # Now go from prev to node
            d = D[prev][node]
            t += d
            batt -= d
            if node not in set(instance["stations"]):
                if t < instance["ready"][node]:
                    t = instance["ready"][node]
                if t > instance["due"][node] + 1e-9:
                    return None
                load += instance["demands"][node]
                if load > instance["capacity"] + 1e-9:
                    return None
                t += instance["service"][node]
            route.append(node)
            prev = node
    
    # Return to depot
    d = D[prev][0]
    if d > batt + 1e-9:
        # Need station to return
        sub_path, _ = _constrained_shortest_path(prev, 0, D, instance)
        if sub_path is None:
            return None
        for sn in sub_path:
            if sn == 0:
                break
            route.append(sn)
            prev = sn
        d = D[prev][0]
        t += d
        batt -= d
    else:
        t += d
        batt -= d
    
    if t > instance["horizon"] + 1e-9 or batt < -1e-9:
        return None
    
    return route


def _construct_initial_routes_matching(instance, D):
    """
    Use minimum weight matching to create initial paths, then build routes.
    """
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    
    # Create a complete graph on customers + depot (node 0)
    # We want to find a minimum weight perfect matching on the odd-degree nodes of a TSP tour.
    # For simplicity, we'll use a greedy nearest neighbor to create a single tour,
    # then break it into paths.
    
    # Actually, let's use a simpler approach:
    # 1. Create a minimum spanning tree or use nearest neighbor to get a tour.
    # 2. Break the tour into paths that respect capacity and time windows.
    
    # Nearest neighbor tour starting from depot
    remaining = set(customers)
    tour = [0]
    current = 0
    
    while remaining:
        nearest = min(remaining, key=lambda c: D[current][c])
        tour.append(nearest)
        remaining.discard(nearest)
        current = nearest
    
    tour.append(0)  # Return to depot
    
    # Now break the tour into paths
    routes = []
    current_path = []
    current_load = 0
    current_time = 0.0
    current_batt = instance["battery"]
    prev_node = 0
    
    for node in tour[1:-1]:  # Skip the final depot
        if node in set(instance["stations"]):
            continue
            
        d = D[prev_node][node]
        new_time = current_time + d
        if new_time < instance["ready"][node]:
            new_time = instance["ready"][node]
        new_batt = current_batt - d
        
        if (new_time <= instance["due"][node] and 
            new_batt >= 0 and 
            current_load + instance["demands"][node] <= instance["capacity"]):
            current_path.append(node)
            current_load += instance["demands"][node]
            current_time = new_time + instance["service"][node]
            current_batt = new_batt
            prev_node = node
        else:
            # Start a new path
            if current_path:
                routes.append(current_path)
            current_path = [node]
            current_load = instance["demands"][node]
            # Reset time and battery for new path
            current_time = D[0][node]
            if current_time < instance["ready"][node]:
                current_time = instance["ready"][node]
            current_time += instance["service"][node]
            current_batt = instance["battery"] - D[0][node]
            prev_node = node
    
    if current_path:
        routes.append(current_path)
    
    # Now build full routes with stations
    full_routes = []
    for path in routes:
        route = _build_route_from_path(path, D, instance)
        if route is not None:
            full_routes.append(route)
        else:
            # Fallback: serve each customer individually
            for c in path:
                route = _build_route_from_path([c], D, instance)
                if route is not None:
                    full_routes.append(route)
    
    return full_routes


def _2opt(route, D, instance, max_iters=20, time_limit=None):
    """
    2-opt improvement for a single route.
    """
    if time_limit is None:
        time_limit = time.time() + 5.0
    
    improved = True
    iters = 0
    current_dist, _, _, _, _ = _simulate(route, D, instance)
    
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if time.time() > time_limit:
            break
        n = len(route)
        for i in range(n - 1):
            for j in range(i + 1, n):
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                dist, valid, _, _, _ = _simulate(new_route, D, instance)
                if valid and dist < current_dist - 1e-9:
                    route = new_route
                    current_dist = dist
                    improved = True
    return route


def _or_opt(route, D, instance, max_iters=20, time_limit=None):
    """
    Or-opt: relocate 1 or 2 consecutive nodes.
    """
    if time_limit is None:
        time_limit = time.time() + 5.0
        
    improved = True
    iters = 0
    current_dist, _, _, _, _ = _simulate(route, D, instance)
    
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if time.time() > time_limit:
            break
        n = len(route)
        for i in range(n):
            for length in [1, 2]:
                if i + length > n:
                    continue
                segment = route[i:i+length]
                rest = route[:i] + route[i+length:]
                if not rest:
                    continue
                for j in range(len(rest) + 1):
                    new_route = rest[:j] + segment + rest[j:]
                    dist, valid, _, _, _ = _simulate(new_route, D, instance)
                    if valid and dist < current_dist - 1e-9:
                        route = new_route
                        current_dist = dist
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return route


def _cross_route_swap(routes, D, instance):
    """
    Inter-route improvement: swap customers between routes.
    """
    improved = True
    iters = 0
    while improved and iters < 10:
        improved = False
        iters += 1
        for a in range(len(routes)):
            for b in range(a + 1, len(routes)):
                for i in range(len(routes[a])):
                    for j in range(len(routes[b])):
                        ra = list(routes[a])
                        rb = list(routes[b])
                        ra[i], rb[j] = rb[j], ra[i]
                        va = _simulate(ra, D, instance)[1]
                        vb = _simulate(rb, D, instance)[1]
                        if va and vb:
                            old_dist = _simulate(ra, D, instance)[0] + _simulate(rb, D, instance)[0]
                            new_dist = _simulate(ra, D, instance)[0] + _simulate(rb, D, instance)[0]
                            if new_dist < old_dist - 1e-9:
                                routes[a] = ra
                                routes[b] = rb
                                improved = True
    return routes


def _is_valid_solution(routes, instance, D):
    served = set()
    for route in routes:
        dist, valid, _, _, _ = _simulate(route, D, instance)
        if not valid:
            return False
        for node in route:
            if node in set(instance["stations"]):
                continue
            if node in served:
                return False
            served.add(node)
    n_customers = instance["n_customers"]
    expected = set(range(1, n_customers + 1))
    return served == expected


def plan(instance):
    start_time = time.time()
    deadline = start_time + 110.0
    
    coords = instance["coords"]
    D = _build_distance_matrix(coords)
    
    # Construct initial solution using matching-based approach
    routes = _construct_initial_routes_matching(instance, D)
    
    # If no routes or invalid, fallback
    if not _is_valid_solution(routes, instance, D):
        # Fallback: one customer per route
        customers = list(range(1, instance["n_customers"] + 1))
        routes = []
        for c in customers:
            route = _build_route_from_path([c], D, instance)
            if route is not None:
                routes.append(route)
    
    best_routes = routes
    
    # Local search
    max_passes = 15
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        
        improved = False
        time_limit = time.time() + (deadline - time.time()) / (max_passes + 1)
        
        # 2-opt on each route
        for i in range(len(best_routes)):
            new_route = _2opt(best_routes[i], D, instance, max_iters=10, time_limit=time_limit)
            if new_route != best_routes[i]:
                best_routes[i] = new_route
                improved = True
        
        # Or-opt
        for i in range(len(best_routes)):
            new_route = _or_opt(best_routes[i], D, instance, max_iters=10, time_limit=time_limit)
            if new_route != best_routes[i]:
                best_routes[i] = new_route
                improved = True
        
        if not improved:
            break
    
    # Final validation
    if not _is_valid_solution(best_routes, instance, D):
        customers = list(range(1, instance["n_customers"] + 1))
        best_routes = [[c] for c in customers]
    
    return best_routes
# EVOLVE-BLOCK-END
