"""EV routing with time windows: constructive + local search with delta evaluation."""
# EVOLVE-BLOCK-START
import numpy as np
import math
import time
import random

# Global deadline guard
_DEADLINE = None
_DEADLINE_MARGIN = 8.0  # seconds before hard limit

def _check_deadline():
    if _DEADLINE is not None and time.time() > _DEADLINE - _DEADLINE_MARGIN:
        return True
    return False


def _simulate_route(route, instance, dist):
    """
    Simulate a route. Returns (feasible, total_distance, final_time, final_battery).
    If infeasible, returns (False, inf, inf, 0).
    """
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    station_set = instance["_station_set"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    total_demand = sum(demands[v] for v in route if v not in station_set)
    if total_demand > capacity:
        return (False, float('inf'), float('inf'), 0.0)

    t = 0.0
    battery = battery_max
    total_dist = 0.0
    prev = 0  # depot

    for v in route:
        d = dist[prev][v]
        t += d / speed
        battery -= d
        total_dist += d

        if battery < -1e-9:
            return (False, float('inf'), float('inf'), 0.0)

        if v in station_set:
            charge_time = (battery_max - battery) * recharge_rate
            t += charge_time
            battery = battery_max
        else:
            if t < ready[v]:
                t = ready[v]
            if t > due[v] + 1e-9:
                return (False, float('inf'), float('inf'), 0.0)
            t += service[v]

        prev = v

    # Return to depot
    d = dist[prev][0]
    t += d / speed
    battery -= d
    total_dist += d

    if battery < -1e-9:
        return (False, float('inf'), float('inf'), 0.0)
    if t > horizon + 1e-9:
        return (False, float('inf'), float('inf'), 0.0)

    return (True, total_dist, t, battery)


def _route_distance(route, dist):
    """Total Euclidean distance including depot legs."""
    if not route:
        return 0.0
    total = dist[0][route[0]]
    for i in range(len(route) - 1):
        total += dist[route[i]][route[i + 1]]
    total += dist[route[-1]][0]
    return total


def _delta_2opt_distance(route, i, j, dist):
    """
    Compute the change in route distance for a 2-opt move reversing route[i:j+1].
    Only considers distance, not feasibility.
    Returns delta (new_dist - old_dist).
    """
    n = len(route)
    # Edges affected: (i-1, i), (i, i+1), (j-1, j), (j, j+1)
    # After reversal: (i-1, j), (j, j+1) stay; (i, j-1) new; (i, i+1) removed
    # Actually for 2-opt reversing segment [i, j]:
    # Old edges: (prev_i, route[i]), (route[i], route[i+1]), ..., (route[j-1], route[j]), (route[j], next_j)
    # New edges: (prev_i, route[j]), (route[j], route[j-1]), ..., (route[i+1], route[i]), (route[i], next_j)
    # The internal edges just get reversed, so their total is the same.
    # Only the two "crossing" edges change:
    # Old: dist(prev_i, route[i]) + dist(route[j], next_j)
    # New: dist(prev_i, route[j]) + dist(route[i], next_j)
    
    prev_i = route[i-1] if i > 0 else 0
    next_j = route[j+1] if j < n-1 else 0
    
    old_d = dist[prev_i][route[i]] + dist[route[j]][next_j]
    new_d = dist[prev_i][route[j]] + dist[route[i]][next_j]
    
    return new_d - old_d


def _solve_instance(instance):
    coords = instance["coords"]
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    stations = instance["stations"]
    demands = instance["demands"]
    due = instance["due"]
    ready = instance["ready"]
    service = instance["service"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    capacity = instance["capacity"]

    # Precompute distances
    n_nodes = len(coords)
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
            dist[i][j] = d
            dist[j][i] = d

    instance["_station_set"] = set(stations)

    # Sort customers by due time
    customers_sorted = sorted(customers, key=lambda v: (due[v], ready[v]))

    # Greedy insertion: build routes
    max_routes = 10
    routes = [[] for _ in range(max_routes)]
    route_loads = [0.0] * max_routes

    if customers_sorted:
        routes[0].append(customers_sorted[0])
        route_loads[0] += demands[customers_sorted[0]]
        unassigned = set(customers_sorted[1:])
    else:
        unassigned = set()

    for cust in customers_sorted:
        if cust not in unassigned:
            continue
        if _check_deadline():
            break
        best_route = -1
        best_cost = float('inf')
        best_pos = -1
        for r in range(len(routes)):
            if route_loads[r] + demands[cust] > capacity:
                continue
            route = routes[r]
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [cust] + route[pos:]
                feasible, d, t, b = _simulate_route(new_route, instance, dist)
                if feasible and d < best_cost:
                    best_cost = d
                    best_route = r
                    best_pos = pos
        if best_route >= 0:
            routes[best_route].insert(best_pos, cust)
            route_loads[best_route] += demands[cust]
            unassigned.discard(cust)
        else:
            new_r = len(routes)
            if new_r < max_routes:
                routes.append([cust])
                route_loads.append(demands[cust])
                unassigned.discard(cust)
            else:
                routes[0].append(cust)
                route_loads[0] += demands[cust]
                unassigned.discard(cust)

    routes = [r for r in routes if r]

    # Local search with delta-based 2-opt and full-sim relocate/swap
    improved = True
    iterations = 0
    max_iters = 500
    while improved and iterations < max_iters:
        improved = False
        iterations += 1
        if _check_deadline():
            break

        # 2-opt within each route using delta evaluation for speed
        for r in range(len(routes)):
            route = routes[r]
            if len(route) < 2:
                continue
            for i in range(len(route) - 1):
                for j in range(i + 1, len(route)):
                    # Quick delta check
                    delta = _delta_2opt_distance(route, i, j, dist)
                    if delta >= -1e-9:
                        continue
                    new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                    f, d, t, b = _simulate_route(new_route, instance, dist)
                    if f:
                        old_d = _route_distance(route, dist)
                        if d < old_d - 1e-9:
                            routes[r] = new_route
                            improved = True

        # Relocate between routes
        for r1 in range(len(routes)):
            if not routes[r1]:
                continue
            for i in range(len(routes[r1])):
                cust = routes[r1][i]
                for r2 in range(len(routes)):
                    if r1 == r2:
                        continue
                    for j in range(len(routes[r2]) + 1):
                        new_r1 = routes[r1][:i] + routes[r1][i + 1:]
                        new_r2 = routes[r2][:j] + [cust] + routes[r2][j:]
                        f1, d1, t1, b1 = _simulate_route(new_r1, instance, dist)
                        f2, d2, t2, b2 = _simulate_route(new_r2, instance, dist)
                        if f1 and f2:
                            old_d = _route_distance(routes[r1], dist) + _route_distance(routes[r2], dist)
                            new_d = _route_distance(new_r1, dist) + _route_distance(new_r2, dist)
                            if new_d < old_d - 1e-9:
                                routes[r1] = new_r1
                                routes[r2] = new_r2
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

        # Swap between routes
        for r1 in range(len(routes)):
            if len(routes[r1]) < 1:
                continue
            for i in range(len(routes[r1])):
                cust1 = routes[r1][i]
                for r2 in range(len(routes)):
                    if r1 == r2:
                        continue
                    for j in range(len(routes[r2])):
                        cust2 = routes[r2][j]
                        if cust1 == cust2:
                            continue
                        new_r1 = routes[r1][:]
                        new_r2 = routes[r2][:]
                        new_r1[i] = cust2
                        new_r2[j] = cust1
                        f1, d1, t1, b1 = _simulate_route(new_r1, instance, dist)
                        f2, d2, t2, b2 = _simulate_route(new_r2, instance, dist)
                        if f1 and f2:
                            old_d = _route_distance(routes[r1], dist) + _route_distance(routes[r2], dist)
                            new_d = _route_distance(new_r1, dist) + _route_distance(new_r2, dist)
                            if new_d < old_d - 1e-9:
                                routes[r1] = new_r1
                                routes[r2] = new_r2
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    # Or-opt: move 1-3 consecutive nodes within or between routes
    for _pass in range(3):
        if _check_deadline():
            break
        improved = False
        for r1 in range(len(routes)):
            if len(routes[r1]) < 2:
                continue
            for i in range(len(routes[r1])):
                for length in [1, 2, 3]:
                    if i + length > len(routes[r1]):
                        continue
                    segment = routes[r1][i:i + length]
                    new_r1 = routes[r1][:i] + routes[r1][i + length:]
                    for r2 in range(len(routes)):
                        for j in range(len(routes[r2]) + 1):
                            if r1 == r2 and j >= i and j <= i + length:
                                continue
                            new_r2 = routes[r2][:j] + segment + routes[r2][j:]
                            f1, d1, t1, b1 = _simulate_route(new_r1, instance, dist)
                            f2, d2, t2, b2 = _simulate_route(new_r2, instance, dist)
                            if f1 and f2:
                                old_d = _route_distance(routes[r1], dist) + _route_distance(routes[r2], dist)
                                new_d = _route_distance(new_r1, dist) + _route_distance(new_r2, dist)
                                if new_d < old_d - 1e-9:
                                    routes[r1] = new_r1
                                    routes[r2] = new_r2
                                    improved = True
                                    break
                        if improved:
                            break
                if improved:
                    break
            if improved:
                break

    routes = [r for r in routes if r]
    return routes


def plan(instance):
    global _DEADLINE
    _DEADLINE = time.time() + 150.0 - 8.0  # 142s effective

    routes = _solve_instance(instance)

    # Verify all customers are served
    served = set()
    for r in routes:
        for v in r:
            if v > 0:
                served.add(v)
    expected = set(range(1, instance["n_customers"] + 1))
    if served != expected:
        routes = [[v] for v in range(1, instance["n_customers"] + 1)]

    return routes
# EVOLVE-BLOCK-END
