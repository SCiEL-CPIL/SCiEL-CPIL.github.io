"""EV routing with time windows – constructive heuristic + 2-opt/Or-opt local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random
import sys

def _euclid(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])

def _simulate_route(route, coords, ready, due, service, n_customers,
                   stations, battery, speed, recharge_rate, horizon, capacity):
    """Return (total_distance, feasible)."""
    total_dist = 0.0
    t = 0.0
    bat = battery
    cur = 0  # depot
    # check capacity
    load = 0
    for v in route:
        if 1 <= v <= n_customers:
            load += 1  # placeholder, real demand passed separately
    # We'll pass demands separately in real calls; here we assume route only customers
    for v in route:
        d = _euclid(coords[cur], coords[v])
        total_dist += d
        t += d / speed
        bat -= d
        if bat < -1e-9:
            return total_dist, False
        if 1 <= v <= n_customers:
            t = max(t, ready[v])
            if t > due[v] + 1e-9:
                return total_dist, False
            t += service[v]
        elif v in stations:
            # charge
            need = battery - bat
            t += need * recharge_rate
            bat = battery
        cur = v
    # return to depot
    d = _euclid(coords[cur], coords[0])
    total_dist += d
    t += d / speed
    bat -= d
    if bat < -1e-9:
        return total_dist, False
    if t > horizon + 1e-9:
        return total_dist, False
    return total_dist, True

def _simulate_full(route, coords, demands, ready, due, service, n_customers,
                   stations, battery, speed, recharge_rate, horizon, capacity):
    """Full simulation with demand check. Returns (dist, feasible)."""
    load = sum(demands[v] for v in route if 1 <= v <= n_customers)
    if load > capacity + 1e-9:
        return 0.0, False
    total_dist = 0.0
    t = 0.0
    bat = battery
    cur = 0
    for v in route:
        d = _euclid(coords[cur], coords[v])
        total_dist += d
        t += d / speed
        bat -= d
        if bat < -1e-9:
            return total_dist, False
        if 1 <= v <= n_customers:
            t = max(t, ready[v])
            if t > due[v] + 1e-9:
                return total_dist, False
            t += service[v]
        elif v in stations:
            need = battery - bat
            t += need * recharge_rate
            bat = battery
        cur = v
    d = _euclid(coords[cur], coords[0])
    total_dist += d
    t += d / speed
    bat -= d
    if bat < -1e-9:
        return total_dist, False
    if t > horizon + 1e-9:
        return total_dist, False
    return total_dist, True

def _constructive(instance):
    """Nearest-neighbor constructive with time-window feasibility."""
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    capacity = instance["capacity"]

    customers = list(range(1, n_customers + 1))
    # Sort by due time
    customers.sort(key=lambda v: due[v])

    routes = []
    unassigned = set(customers)
    n = len(coords)

    # Precompute distances
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])

    while unassigned:
        route = []
        cur = 0
        t = 0.0
        bat = battery
        load = 0
        remaining = set(unassigned)

        while remaining:
            best = None
            best_cost = float('inf')
            for v in remaining:
                d = dist[cur][v]
                new_t = t + d / speed
                new_bat = bat - d
                if new_bat < -1e-9:
                    continue
                if 1 <= v <= n_customers:
                    if load + demands[v] > capacity + 1e-9:
                        continue
                    arrival = max(new_t, ready[v])
                    if arrival > due[v] + 1e-9:
                        continue
                    # Check if we can still return
                    return_d = dist[v][0]
                    final_t = arrival + service[v] + return_d / speed
                    if final_t > horizon + 1e-9:
                        continue
                    if new_bat - return_d < -1e-9:
                        continue
                    cost = d
                else:
                    # station
                    new_t2 = new_t + (battery - new_bat) * recharge_rate
                    # check return
                    return_d = dist[v][0]
                    final_t = new_t2 + return_d / speed
                    if final_t > horizon + 1e-9:
                        continue
                    if new_bat - return_d < -1e-9:
                        continue
                    cost = d

                if cost < best_cost:
                    best_cost = cost
                    best = v

            if best is None:
                break
            d = dist[cur][best]
            t += d / speed
            bat -= d
            if 1 <= best <= n_customers:
                t = max(t, ready[best])
                t += service[best]
                load += demands[best]
                route.append(best)
                remaining.discard(best)
            else:
                t += (battery - bat) * recharge_rate
                bat = battery
                route.append(best)
            cur = best

        if route:
            # Verify feasibility
            dist_total, feas = _simulate_full(route, coords, demands, ready, due,
                                             service, n_customers, stations,
                                             battery, speed, recharge_rate,
                                             horizon, capacity)
            if feas:
                routes.append(route)
                unassigned -= set(route)
            else:
                # Fallback: serve one customer at a time
                for v in list(unassigned):
                    single = [v]
                    dist_total, feas = _simulate_full(single, coords, demands, ready, due,
                                                     service, n_customers, stations,
                                                     battery, speed, recharge_rate,
                                                     horizon, capacity)
                    if feas:
                        routes.append(single)
                        unassigned.discard(v)
                break
        else:
            # Fallback: serve one customer at a time
            for v in list(unassigned):
                single = [v]
                dist_total, feas = _simulate_full(single, coords, demands, ready, due,
                                                 service, n_customers, stations,
                                                 battery, speed, recharge_rate,
                                                 horizon, capacity)
                if feas:
                    routes.append(single)
                    unassigned.discard(v)
            break

    # If any unassigned remain, force single-route
    for v in list(unassigned):
        routes.append([v])

    return routes

def _route_distance(route, coords):
    total = 0.0
    cur = 0
    for v in route:
        total += math.hypot(coords[cur][0] - coords[v][0], coords[cur][1] - coords[v][1])
        cur = v
    total += math.hypot(coords[cur][0] - coords[0], coords[cur][1] - coords[0])
    return total

def _local_search(routes, instance, time_limit=100.0):
    """2-opt and Or-opt local search with feasibility checks."""
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    capacity = instance["capacity"]

    n = len(coords)
    # Precompute distance matrix
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])

    def is_feasible(route):
        return _simulate_full(route, coords, demands, ready, due, service,
                             n_customers, stations, battery, speed, recharge_rate,
                             horizon, capacity)[1]

    def total_dist():
        return sum(_route_distance(r, coords) for r in routes)

    best = total_dist()
    start = time.time()
    improved = True
    max_iters = 500

    while improved and time.time() - start < time_limit:
        improved = False
        for it in range(max_iters):
            if time.time() - start > time_limit:
                break
            for ri in range(len(routes)):
                route = routes[ri]
                L = len(route)
                if L < 2:
                    continue
                # 2-opt: try reversing segment [i, j]
                for i in range(L - 1):
                    for j in range(i + 1, L):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        if is_feasible(new_route):
                            new_total = total_dist() - _route_distance(route, coords) + _route_distance(new_route, coords)
                            if new_total < best - 1e-9:
                                routes[ri] = new_route
                                best = new_total
                                improved = True
                # Or-opt: try moving segment of length 1 or 2
                for seg_len in [1, 2]:
                    for i in range(L - seg_len + 1):
                        seg = route[i:i+seg_len]
                        rest = route[:i] + route[i+seg_len:]
                        for j in range(len(rest) + 1):
                            new_route = rest[:j] + seg + rest[j:]
                            if is_feasible(new_route):
                                new_total = total_dist() - _route_distance(route, coords) + _route_distance(new_route, coords)
                                if new_total < best - 1e-9:
                                    routes[ri] = new_route
                                    best = new_total
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        # After each full pass, try merging routes
        if time.time() - start > time_limit:
            break
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                if time.time() - start > time_limit:
                    break
                merged = routes[i] + routes[j]
                if is_feasible(merged):
                    new_total = total_dist() - _route_distance(routes[i], coords) - _route_distance(routes[j], coords) + _route_distance(merged, coords)
                    if new_total < best - 1e-9:
                        routes[i] = merged
                        routes.pop(j)
                        best = new_total
                        improved = True
                        break
            if improved:
                break

    return routes

def plan(instance):
    start = time.time()
    deadline = 140.0  # safety margin before 150s kill

    # Constructive phase
    routes = _constructive(instance)

    # Local search phase
    remaining = deadline - (time.time() - start)
    if remaining > 5.0:
        routes = _local_search(routes, instance, time_limit=remaining - 5.0)

    # Final validation
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    capacity = instance["capacity"]

    served = set()
    valid_routes = []
    for route in routes:
        dist_total, feas = _simulate_full(route, coords, demands, ready, due,
                                         service, n_customers, stations,
                                         battery, speed, recharge_rate,
                                         horizon, capacity)
        if feas:
            for v in route:
                if 1 <= v <= n_customers:
                    served.add(v)
            valid_routes.append(route)
        else:
            # Fallback: split into singles
            for v in route:
                if 1 <= v <= n_customers and v not in served:
                    single = [v]
                    _, feas2 = _simulate_full(single, coords, demands, ready, due,
                                             service, n_customers, stations,
                                             battery, speed, recharge_rate,
                                             horizon, capacity)
                    if feas2:
                        valid_routes.append(single)
                        served.add(v)

    # Ensure all customers served
    for v in range(1, n_customers + 1):
        if v not in served:
            valid_routes.append([v])
            served.add(v)

    return valid_routes
# EVOLVE-BLOCK-END
