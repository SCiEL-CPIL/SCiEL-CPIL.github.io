"""EV routing with time windows – Clarke-Wright savings + fast local search."""
# EVOLVE-BLOCK-START
import math
import time
import random
import numpy as np
from heapq import heappush, heappop


def _build_dist_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        xi, yi = coords[i][0], coords[i][1]
        for j in range(i + 1, n):
            d = math.hypot(xi - coords[j][0], yi - coords[j][1])
            D[i, j] = d
            D[j, i] = d
    return D


def _simulate(route, instance, D):
    """Simulate a single route. Returns (total_distance, valid, end_time, end_battery)."""
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
    t = 0.0
    battery = battery_max
    load = 0
    total_dist = 0.0
    prev = depot

    for node in route:
        d = D[prev][node]
        t += d / speed
        battery -= d
        total_dist += d

        if battery < -1e-9:
            return total_dist, False, t, battery

        if node in stations:
            # Charging station: recharge
            energy = battery_max - battery
            t += energy * recharge_rate
            battery = battery_max
        else:
            # Customer node
            if node > n_customers:
                return total_dist, False, t, battery
            # Time window
            if t < ready[node]:
                t = ready[node]
            if t > due[node] + 1e-9:
                return total_dist, False, t, battery
            # Service
            t += service[node]
            # Load
            load += demands[node]
            if load > capacity + 1e-9:
                return total_dist, False, t, battery

        prev = node

    # Return to depot
    d = D[prev][depot]
    t += d / speed
    battery -= d
    total_dist += d

    if battery < -1e-9:
        return total_dist, False, t, battery
    if t > horizon + 1e-9:
        return total_dist, False, t, battery

    return total_dist, True, t, battery


def _total_distance(routes, D):
    total = 0.0
    depot = 0
    for route in routes:
        prev = depot
        for node in route:
            total += D[prev][node]
            prev = node
        total += D[prev][depot]
    return total


def _is_valid_solution(routes, instance, D):
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
        _, valid, _, _ = _simulate(route, instance, D)
        if not valid:
            return False
    return True


def _clarke_wright_savings(instance, D):
    """Clarke-Wright savings heuristic for initial route construction."""
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
    route_of = {i: i - 1 for i in range(1, n_customers + 1)}  # customer -> route index

    # Compute savings
    savings = []
    for i in range(1, n_customers + 1):
        for j in range(i + 1, n_customers + 1):
            s = D[depot][i] + D[depot][j] - D[i][j]
            savings.append((s, i, j))
    savings.sort(reverse=True)

    # Try to merge routes
    for s, i, j in savings:
        if s <= 0:
            break
        ri = route_of[i]
        rj = route_of[j]
        if ri == rj:
            continue

        route_i = routes[ri]
        route_j = routes[rj]

        # Check if i is at end of route_i and j is at start of route_j
        # or i is at start and j is at end, etc.
        # For simplicity, only merge if i is last in route_i and j is first in route_j
        # or i is first in route_i and j is last in route_j

        can_merge = False
        new_route = None

        if route_i[-1] == i and route_j[0] == j:
            # i at end of route_i, j at start of route_j
            candidate = route_i + route_j
            # Check feasibility
            _, valid, _, _ = _simulate(candidate, instance, D)
            if valid:
                can_merge = True
                new_route = candidate
        elif route_i[0] == i and route_j[-1] == j:
            # i at start of route_i, j at end of route_j
            candidate = route_j + route_i
            _, valid, _, _ = _simulate(candidate, instance, D)
            if valid:
                can_merge = True
                new_route = candidate
        elif route_i[-1] == i and route_j[-1] == j:
            # i at end of route_i, j at end of route_j
            candidate = route_i + route_j[::-1]
            _, valid, _, _ = _simulate(candidate, instance, D)
            if valid:
                can_merge = True
                new_route = candidate
        elif route_i[0] == i and route_j[0] == j:
            # i at start of route_i, j at start of route_j
            candidate = route_i[::-1] + route_j
            _, valid, _, _ = _simulate(candidate, instance, D)
            if valid:
                can_merge = True
                new_route = candidate

        if can_merge:
            # Merge routes
            routes[ri] = new_route
            routes[rj] = []
            for node in new_route:
                route_of[node] = ri

    # Remove empty routes
    routes = [r for r in routes if r]
    return routes


def _2opt(route, instance, D, max_iters=50):
    """2-opt local search within a single route."""
    n = len(route)
    if n < 2:
        return route

    best = list(route)
    best_dist = _total_distance([best], D)

    for _ in range(max_iters):
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                if j == i + 1:
                    continue
                # Try reversing segment i..j
                new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                _, valid, _, _ = _simulate(new_route, instance, D)
                if not valid:
                    continue
                d = _total_distance([new_route], D)
                if d < best_dist - 1e-9:
                    best = new_route
                    best_dist = d
                    improved = True
        if not improved:
            break

    return best


def _or_opt(routes, instance, D, max_iters=30):
    """Or-opt: move 1-3 consecutive customers to another position."""
    n_routes = len(routes)
    best_routes = [list(r) for r in routes]
    best_dist = _total_distance(best_routes, D)

    for _ in range(max_iters):
        improved = False
        for ri in range(n_routes):
            route = best_routes[ri]
            n = len(route)
            for i in range(n):
                for length in [1, 2, 3]:
                    if i + length > n:
                        continue
                    segment = route[i:i+length]
                    rest = route[:i] + route[i+length:]
                    if not rest:
                        continue
                    for j in range(len(rest) + 1):
                        new_route = rest[:j] + segment + rest[j:]
                        new_routes = [best_routes[k] if k != ri else new_route for k in range(n_routes)]
                        if not _is_valid_solution(new_routes, instance, D):
                            continue
                        d = _total_distance(new_routes, D)
                        if d < best_dist - 1e-9:
                            best_routes = new_routes
                            best_dist = d
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break

    return best_routes


def _cross_route_swap(routes, instance, D, max_iters=20):
    """Swap customers between different routes."""
    n_routes = len(routes)
    best_routes = [list(r) for r in routes]
    best_dist = _total_distance(best_routes, D)

    for _ in range(max_iters):
        improved = False
        for ri in range(n_routes):
            for rj in range(ri + 1, n_routes):
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
                        new_routes = [best_routes[k] if k != ri else new_ri for k in range(n_routes)]
                        new_routes[rj] = new_rj
                        if not _is_valid_solution(new_routes, instance, D):
                            continue
                        d = _total_distance(new_routes, D)
                        if d < best_dist - 1e-9:
                            best_routes = new_routes
                            best_dist = d
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break

    return best_routes


def _station_opt(routes, instance, D, max_iters=20):
    """Try inserting charging stations at various positions to reduce travel distance."""
    n_routes = len(routes)
    best_routes = [list(r) for r in routes]
    best_dist = _total_distance(best_routes, D)
    stations = instance["stations"]
    
    for _ in range(max_iters):
        improved = False
        for ri in range(n_routes):
            route = best_routes[ri]
            n = len(route)
            # Try inserting a station between each pair of consecutive nodes
            for i in range(n):
                for station in stations:
                    if station in route:
                        continue
                    new_route = route[:i+1] + [station] + route[i+1:]
                    new_routes = [best_routes[k] if k != ri else new_route for k in range(n_routes)]
                    if not _is_valid_solution(new_routes, instance, D):
                        continue
                    d = _total_distance(new_routes, D)
                    if d < best_dist - 1e-9:
                        best_routes = new_routes
                        best_dist = d
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return best_routes


def plan(instance):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    n_customers = instance["n_customers"]
    D = _build_dist_matrix(instance["coords"])

    # Generate initial solution using Clarke-Wright savings
    routes = _clarke_wright_savings(instance, D)

    if not _is_valid_solution(routes, instance, D):
        # Fallback: one customer per route
        customers = list(range(1, n_customers + 1))
        customers.sort(key=lambda v: instance["due"][v])
        routes = [[v] for v in customers]

    best_routes = [list(r) for r in routes]
    best_dist = _total_distance(best_routes, D)

    # Apply local search
    if time.time() < deadline - 10:
        for route_idx in range(len(best_routes)):
            best_routes[route_idx] = _2opt(best_routes[route_idx], instance, D, max_iters=30)
        if _is_valid_solution(best_routes, instance, D):
            d = _total_distance(best_routes, D)
            if d < best_dist:
                best_dist = d

    if time.time() < deadline - 10:
        best_routes = _or_opt(best_routes, instance, D, max_iters=20)
        if _is_valid_solution(best_routes, instance, D):
            d = _total_distance(best_routes, D)
            if d < best_dist:
                best_dist = d

    if time.time() < deadline - 10:
        best_routes = _cross_route_swap(best_routes, instance, D, max_iters=10)
        if _is_valid_solution(best_routes, instance, D):
            d = _total_distance(best_routes, D)
            if d < best_dist:
                best_dist = d

    # Additional passes if time remains
    while time.time() < deadline - 5:
        improved = False
        for route_idx in range(len(best_routes)):
            new_route = _2opt(best_routes[route_idx], instance, D, max_iters=20)
            if new_route != best_routes[route_idx]:
                best_routes[route_idx] = new_route
                if _is_valid_solution(best_routes, instance, D):
                    d = _total_distance(best_routes, D)
                    if d < best_dist - 1e-9:
                        best_dist = d
                        improved = True

        if time.time() < deadline - 5:
            new_routes = _or_opt(best_routes, instance, D, max_iters=10)
            if _is_valid_solution(new_routes, instance, D):
                d = _total_distance(new_routes, D)
                if d < best_dist - 1e-9:
                    best_routes = new_routes
                    best_dist = d
                    improved = True

        if time.time() < deadline - 5:
            new_routes = _cross_route_swap(best_routes, instance, D, max_iters=5)
            if _is_valid_solution(new_routes, instance, D):
                d = _total_distance(new_routes, D)
                if d < best_dist - 1e-9:
                    best_routes = new_routes
                    best_dist = d
                    improved = True

        if time.time() < deadline - 5:
            new_routes = _station_opt(best_routes, instance, D, max_iters=5)
            if _is_valid_solution(new_routes, instance, D):
                d = _total_distance(new_routes, D)
                if d < best_dist - 1e-9:
                    best_routes = new_routes
                    best_dist = d
                    improved = True

        if not improved:
            break

    return best_routes
# EVOLVE-BLOCK-END
