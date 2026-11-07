"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.spatial.distance import cdist


def _dist_matrix(coords):
    n = len(coords)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            d[i, j] = math.hypot(dx, dy)
            d[j, i] = d[i, j]
    return d


def _route_cost(route, dist, instance):
    """Compute total distance and check feasibility of a route."""
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

    total_dist = 0.0
    total_demand = 0
    battery = battery_max
    t = 0.0
    prev = 0  # depot

    for node in route:
        d = dist[prev][node]
        total_dist += d
        battery -= d
        t += d / speed
        if battery < 0:
            return None, None  # infeasible

        if node in stations:
            # Recharge
            energy_needed = battery_max - battery
            recharge_time = energy_needed * recharge_rate
            t += recharge_time
            battery = battery_max
        else:
            # Customer service
            if t < ready[node]:
                t = ready[node]
            if t > due[node]:
                return None, None  # infeasible
            t += service[node]
            total_demand += demands[node]
            if total_demand > capacity:
                return None, None  # infeasible

        prev = node

    # Return to depot
    d = dist[prev][0]
    total_dist += d
    battery -= d
    t += d / speed
    if battery < 0 or t > horizon:
        return None, None  # infeasible

    return total_dist, total_demand


def _solve_instance(instance, time_budget=140.0):
    start_time = time.time()
    coords = instance["coords"]
    n = len(coords)
    customers = list(range(1, instance["n_customers"] + 1))
    stations = set(instance["stations"])
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    dist = _dist_matrix(coords)

    # Initial solution: sort by due time, one customer per route
    customers_sorted = sorted(customers, key=lambda v: due[v])
    initial_routes = [[v] for v in customers_sorted]

    # Check initial solution validity
    best_routes = initial_routes
    best_cost = sum(_route_cost(r, dist, instance)[0] for r in initial_routes if _route_cost(r, dist, instance)[0] is not None)

    def total_cost(routes):
        total = 0.0
        for r in routes:
            c, _ = _route_cost(r, dist, instance)
            if c is None:
                return float('inf')
            total += c
        return total

    best_cost = total_cost(best_routes)

    # 2-opt improvement
    def two_opt(routes, max_iter=1000):
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            for ri in range(len(routes)):
                route = routes[ri]
                if len(route) < 2:
                    continue
                # Try reversing segments
                for i in range(len(route)):
                    for j in range(i + 2, len(route) + 1):
                        new_route = route[:i] + list(reversed(route[i:j])) + route[j:]
                        new_routes = routes[:ri] + [new_route] + routes[ri+1:]
                        new_cost = total_cost(new_routes)
                        if new_cost < best_cost - 1e-9:
                            routes = new_routes
                            best_cost = new_cost
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            # Check time
            if time.time() - start_time > time_budget - 5:
                break
        return routes, best_cost

    # OR-opt: move 1 or 2 consecutive customers to another position
    def or_opt(routes, max_iter=1000):
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            for ri in range(len(routes)):
                route = routes[ri]
                if len(route) < 2:
                    continue
                for i in range(len(route)):
                    for length in [1, 2]:
                        if i + length > len(route):
                            continue
                        segment = route[i:i+length]
                        rest = route[:i] + route[i+length:]
                        if not rest:
                            continue
                        for pos in range(len(rest) + 1):
                            new_route = rest[:pos] + segment + rest[pos:]
                            new_routes = routes[:ri] + [new_route] + routes[ri+1:]
                            new_cost = total_cost(new_routes)
                            if new_cost < best_cost - 1e-9:
                                routes = new_routes
                                best_cost = new_cost
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if time.time() - start_time > time_budget - 5:
                break
        return routes, best_cost

    # Apply local search
    routes, cost = two_opt(best_routes, max_iter=200)
    best_routes, best_cost = routes, cost
    routes, cost = or_opt(best_routes, max_iter=200)
    best_routes, best_cost = routes, cost

    # Repeat local search if time remains
    while time.time() - start_time < time_budget - 10:
        routes, cost = two_opt(best_routes, max_iter=100)
        if cost < best_cost - 1e-9:
            best_routes, best_cost = routes, cost
        routes, cost = or_opt(best_routes, max_iter=100)
        if cost < best_cost - 1e-9:
            best_routes, best_cost = routes, cost
        if time.time() - start_time > time_budget - 5:
            break

    return best_routes


def plan(instance):
    # Solve the single instance (the task says 3 fixed generated instances, but plan is called per instance)
    routes = _solve_instance(instance, time_budget=115.0)
    return routes
# EVOLVE-BLOCK-END
