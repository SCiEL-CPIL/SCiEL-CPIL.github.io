"""EV routing with time windows – multi-route construction + local search."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np
from scipy.spatial.distance import cdist


def _euclid(coords, i, j):
    return math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])


def _build_distance_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
            D[i, j] = d
            D[j, i] = d
    return D


def _simulate(route, D, instance, speed=1.0):
    """
    Simulate a single route (list of customer/station node ids, depot implicit at both ends).
    Returns (total_distance, valid, final_time, final_battery, load).
    """
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations_set = set(instance["stations"])

    n = len(coords)
    t = 0.0
    batt = battery_max
    load = 0
    dist_total = 0.0
    prev = 0  # depot

    for node in route:
        d = D[prev][node]
        dist_total += d
        t += d / speed
        batt -= d
        if batt < -1e-9:
            return dist_total, False, t, batt, load
        if node in stations_set:
            # recharge: time added = (battery_max - batt) * recharge_rate
            recharge_time = (battery_max - batt) * recharge_rate
            t += recharge_time
            batt = battery_max
        else:
            # customer
            if t < ready[node]:
                t = ready[node]
            if t > due[node] + 1e-9:
                return dist_total, False, t, batt, load
            load += demands[node]
            if load > capacity + 1e-9:
                return dist_total, False, t, batt, load
            t += service[node]
        prev = node

    # return to depot
    d = D[prev][0]
    dist_total += d
    t += d / speed
    batt -= d
    if batt < -1e-9:
        return dist_total, False, t, batt, load
    if t > horizon + 1e-9:
        return dist_total, False, t, batt, load

    return dist_total, True, t, batt, load


def _total_distance(routes, D):
    total = 0.0
    for route in routes:
        if not route:
            continue
        prev = 0
        for node in route:
            total += D[prev][node]
            prev = node
        total += D[prev][0]
    return total


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


def _construct_initial_routes(instance, D, seed=0):
    """
    Greedy nearest-neighbour with time-window and resource feasibility.
    Returns a list of routes (each a list of node ids, customers only; stations inserted later if needed).
    """
    rng = random.Random(seed)
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = instance["stations"]
    stations_set = set(stations)
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    rng.shuffle(customers)

    routes = []
    remaining = set(customers)

    while remaining:
        route = []
        t = 0.0
        batt = battery_max
        load = 0
        prev = 0

        while remaining:
            best_node = None
            best_dist = float('inf')
            for c in remaining:
                d = D[prev][c]
                # check battery
                new_batt = batt - d
                if new_batt < -1e-9:
                    continue
                # check time window
                new_t = t + d
                if new_t < ready[c]:
                    new_t = ready[c]
                if new_t > due[c]:
                    continue
                # check capacity
                new_load = load + demands[c]
                if new_load > capacity:
                    continue
                # check horizon feasibility (rough: go to c, serve, return)
                ret_d = D[c][0]
                final_t = new_t + service[c] + ret_d
                final_batt = new_batt - ret_d
                if final_batt < -1e-9 or final_t > horizon:
                    continue
                if d < best_dist:
                    best_dist = d
                    best_node = c

            if best_node is None:
                # try inserting a station to recharge
                # find the customer that is closest but battery-insufficient, insert station before it
                # For simplicity, just break and start a new route
                break

            # move to best_node
            d = D[prev][best_node]
            t += d
            batt -= d
            if t < ready[best_node]:
                t = ready[best_node]
            t += service[best_node]
            load += demands[best_node]
            route.append(best_node)
            remaining.discard(best_node)
            prev = best_node

        if route:
            routes.append(route)
        else:
            # force-assign one customer to avoid infinite loop
            c = min(remaining, key=lambda x: D[0][x])
            routes.append([c])
            remaining.discard(c)

    return routes


def _try_insert_station(route, instance, D, pos, target_node):
    """
    Try to insert a station at position pos in the route so that target_node (at pos) becomes feasible.
    Returns new route if successful, else None.
    """
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = instance["stations"]

    # Try each station
    for st in stations:
        new_route = route[:pos] + [st] + route[pos:]
        dist, valid, _, _, _ = _simulate(new_route, D, instance)
        if valid:
            return new_route
    return None


def _fix_route(route, instance, D, max_iters=20):
    """
    Attempt to fix a route by inserting stations where needed.
    Returns fixed route or None.
    """
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = instance["stations"]
    stations_set = set(stations)

    for _ in range(max_iters):
        dist, valid, t, batt, load = _simulate(route, D, instance)
        if valid:
            return route

        # Find first infeasible point
        t_sim = 0.0
        batt_sim = battery_max
        load_sim = 0
        prev = 0
        for i, node in enumerate(route):
            d = D[prev][node]
            t_sim += d
            batt_sim -= d
            if batt_sim < -1e-9:
                # Need to insert station before this node
                new_route = _try_insert_station(route, instance, D, i, node)
                if new_route is not None:
                    route = new_route
                    break
                else:
                    return None
            if node not in stations_set:
                if t_sim < ready[node]:
                    t_sim = ready[node]
                if t_sim > due[node]:
                    # Time window violation – hard to fix by station insertion alone
                    return None
                t_sim += service[node]
                load_sim += demands[node]
                if load_sim > capacity:
                    return None
            prev = node

        # If we get here, loop again
        continue

    return route if _simulate(route, D, instance)[1] else None


def _2opt(route, D, instance, max_iters=10, time_limit=None):
    """
    2-opt improvement for a single route, respecting feasibility.
    Optimized for speed with early termination.
    """
    if time_limit is None:
        time_limit = time.time() + 5.0 # 5 sec hard limit per route? No, global.
    
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
                # Reverse segment i..j
                new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                dist, valid, _, _, _ = _simulate(new_route, D, instance)
                if valid and dist < current_dist - 1e-9:
                    route = new_route
                    current_dist = dist
                    improved = True
    return route


def _or_opt(route, D, instance, max_iters=10, time_limit=None):
    """
    Or-opt: relocate 1 or 2 consecutive nodes.
    Optimized for speed.
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
                segment = route[i:i + length]
                rest = route[:i] + route[i + length:]
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


def _cross_route_2opt(routes, D, instance):
    """
    Simple inter-route improvement: swap customers between routes if it reduces total distance.
    """
    improved = True
    iters = 0
    while improved and iters < 20:
        improved = False
        iters += 1
        for a in range(len(routes)):
            for b in range(a + 1, len(routes)):
                for i in range(len(routes[a])):
                    for j in range(len(routes[b])):
                        # Swap routes[a][i] and routes[b][j]
                        ra = list(routes[a])
                        rb = list(routes[b])
                        ra[i], rb[j] = rb[j], ra[i]
                        # Check validity
                        va = _simulate(ra, D, instance)[1]
                        vb = _simulate(rb, D, instance)[1]
                        if va and vb:
                            old_dist = _total_distance([ra, rb], D)
                            new_dist = _simulate(ra, D, instance)[0] + _simulate(rb, D, instance)[0]
                            if new_dist < old_dist - 1e-9:
                                routes[a] = ra
                                routes[b] = rb
                                improved = True
    return routes


def plan(instance):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin for 120s limit

    coords = instance["coords"]
    D = _build_distance_matrix(coords)

    # Construct initial solution
    routes = _construct_initial_routes(instance, D, seed=42)

    # Fix routes by inserting stations
    fixed_routes = []
    for r in routes:
        fixed = _fix_route(r, instance, D)
        if fixed is not None:
            fixed_routes.append(fixed)
        else:
            # Fallback: serve this customer alone in a new route
            for c in r:
                fixed_routes.append([c])

    best_routes = fixed_routes

    # Local Search Loop
    # We limit the number of passes to ensure we finish in time.
    # Each pass does 2-opt and Or-opt on all routes.
    max_passes = 20
    for _ in range(max_passes):
        if time.time() > deadline:
            break
            
        improved = False
        current_time_limit = time.time() + (deadline - time.time()) / (max_passes - _ + 1)
        
        # 2-opt on each route
        for i in range(len(best_routes)):
            new_route = _2opt(best_routes[i], D, instance, max_iters=10, time_limit=current_time_limit)
            if new_route != best_routes[i]:
                best_routes[i] = new_route
                improved = True

        # Or-opt
        for i in range(len(best_routes)):
            new_route = _or_opt(best_routes[i], D, instance, max_iters=10, time_limit=current_time_limit)
            if new_route != best_routes[i]:
                best_routes[i] = new_route
                improved = True

        if not improved:
            break

    # Final validation
    if not _is_valid_solution(best_routes, instance, D):
        # Fallback to one-customer-per-route
        customers = list(range(1, instance["n_customers"] + 1))
        best_routes = [[c] for c in customers]

    return best_routes
# EVOLVE-BLOCK-END
