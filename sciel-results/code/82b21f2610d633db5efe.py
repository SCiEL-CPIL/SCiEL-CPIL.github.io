"""Rewritten program: EV routing with time windows, battery, and charging stations."""
# EVOLVE-BLOCK-START
import math
import time
import random
import numpy as np


def _euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _route_cost(route, inst, coords, dist_mat):
    """Return total distance for a route (depot implicit at both ends)."""
    if not route:
        return 0.0
    total = 0.0
    prev = 0
    for node in route:
        total += dist_mat[prev][node]
        prev = node
    total += dist_mat[prev][0]
    return total


def _simulate_route(route, inst, coords, dist_mat):
    """
    Simulate a route. Returns (feasible, total_time, total_distance).
    Checks: capacity, time windows, battery, horizon.
    """
    capacity = inst["capacity"]
    battery_max = inst["battery"]
    speed = inst["speed"]
    recharge_rate = inst["recharge_rate"]
    horizon = inst["horizon"]
    demands = inst["demands"]
    ready = inst["ready"]
    due = inst["due"]
    service = inst["service"]
    stations = set(inst["stations"])

    # Check capacity
    total_demand = sum(demands[v] for v in route if v not in stations)
    if total_demand > capacity:
        return False, 0, 0

    t = 0.0
    battery = battery_max
    total_dist = 0.0
    prev = 0

    for node in route:
        d = dist_mat[prev][node]
        t += d / speed
        battery -= d

        if battery < -1e-9:
            return False, 0, 0

        if node in stations:
            # Charging station: add recharge time, refill battery
            energy_needed = battery_max - battery
            t += energy_needed * recharge_rate
            battery = battery_max
        else:
            # Customer: check time window
            if t < ready[node]:
                t = ready[node]
            if t > due[node] + 1e-9:
                return False, 0, 0
            t += service[node]

        total_dist += d
        prev = node

    # Return to depot
    d_back = dist_mat[prev][0]
    t += d_back / speed
    battery -= d_back
    total_dist += d_back

    if battery < -1e-9:
        return False, 0, 0
    if t > horizon + 1e-9:
        return False, 0, 0

    return True, t, total_dist


def _greedy_construct(inst, coords, dist_mat):
    """
    Build initial routes using a greedy nearest-neighbor with feasibility checks.
    Each customer is assigned to the best existing route or a new route.
    """
    n_cust = inst["n_customers"]
    customers = list(range(1, n_cust + 1))
    due = inst["due"]
    ready = inst["ready"]
    # Sort by due time (earliest first)
    customers.sort(key=lambda v: due[v])

    routes = []

    for cust in customers:
        best_route_idx = -1
        best_pos = -1
        best_delta_dist = float('inf')

        # Try inserting into each existing route
        for ri, route in enumerate(routes):
            # Try each position in the route
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [cust] + route[pos:]
                feasible, _, _ = _simulate_route(new_route, inst, coords, dist_mat)
                if feasible:
                    # Compute delta distance
                    old_dist = _route_cost(route, inst, coords, dist_mat)
                    new_dist = _route_cost(new_route, inst, coords, dist_mat)
                    delta = new_dist - old_dist
                    if delta < best_delta_dist:
                        best_delta_dist = delta
                        best_route_idx = ri
                        best_pos = pos

        # If no feasible insertion, try a new route
        if best_route_idx == -1:
            new_route = [cust]
            feasible, _, _ = _simulate_route(new_route, inst, coords, dist_mat)
            if feasible:
                routes.append(new_route)
                continue
            else:
                # Shouldn't happen with single customer, but skip if infeasible
                continue

        # Insert into best route
        routes[best_route_idx].insert(best_pos, cust)

    return routes


def _two_opt(route, inst, coords, dist_mat):
    """Apply 2-opt moves within a route. Returns improved route."""
    n = len(route)
    if n < 2:
        return route
    improved = True
    best = route[:]
    best_cost = _route_cost(best, inst, coords, dist_mat)

    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Reverse segment i..j
                new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                feasible, _, _ = _simulate_route(new_route, inst, coords, dist_mat)
                if feasible:
                    new_cost = _route_cost(new_route, inst, coords, dist_mat)
                    if new_cost < best_cost - 1e-9:
                        best = new_route
                        best_cost = new_cost
                        improved = True
    return best


def _or_opt(route, inst, coords, dist_mat, segment_len=3):
    """Apply Or-opt moves: relocate a segment of up to segment_len nodes."""
    n = len(route)
    if n < 2:
        return route
    improved = True
    best = route[:]
    best_cost = _route_cost(best, inst, coords, dist_mat)

    while improved:
        improved = False
        for seg in range(1, min(segment_len, n - 1) + 1):
            for i in range(n - seg + 1):
                segment = best[i:i+seg]
                remaining = best[:i] + best[i+seg:]
                if not remaining:
                    continue
                for j in range(len(remaining) + 1):
                    new_route = remaining[:j] + segment + remaining[j:]
                    if new_route == best:
                        continue
                    feasible, _, _ = _simulate_route(new_route, inst, coords, dist_mat)
                    if feasible:
                        new_cost = _route_cost(new_route, inst, coords, dist_mat)
                        if new_cost < best_cost - 1e-9:
                            best = new_route
                            best_cost = new_cost
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    return best


def _inter_route_moves(routes, inst, coords, dist_mat):
    """
    Try relocate and swap moves between routes.
    Returns improved list of routes.
    """
    improved = True
    while improved:
        improved = False
        for i in range(len(routes)):
            for j in range(len(routes)):
                if i == j:
                    continue
                # Relocate: move one node from route i to route j
                for pos_i in range(len(routes[i])):
                    node = routes[i][pos_i]
                    remaining_i = routes[i][:pos_i] + routes[i][pos_i+1:]
                    if not remaining_i:
                        continue
                    for pos_j in range(len(routes[j]) + 1):
                        new_j = routes[j][:pos_j] + [node] + routes[j][pos_j:]
                        feasible_i, _, _ = _simulate_route(remaining_i, inst, coords, dist_mat)
                        feasible_j, _, _ = _simulate_route(new_j, inst, coords, dist_mat)
                        if feasible_i and feasible_j:
                            old_cost = _route_cost(routes[i], inst, coords, dist_mat) + _route_cost(routes[j], inst, coords, dist_mat)
                            new_cost = _route_cost(remaining_i, inst, coords, dist_mat) + _route_cost(new_j, inst, coords, dist_mat)
                            if new_cost < old_cost - 1e-9:
                                routes[i] = remaining_i
                                routes[j] = new_j
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        # Also try swap: exchange two nodes between different routes
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                for pos_i in range(len(routes[i])):
                    for pos_j in range(len(routes[j])):
                        node_i = routes[i][pos_i]
                        node_j = routes[j][pos_j]
                        new_i = routes[i][:]
                        new_j = routes[j][:]
                        new_i[pos_i] = node_j
                        new_j[pos_j] = node_i
                        feasible_i, _, _ = _simulate_route(new_i, inst, coords, dist_mat)
                        feasible_j, _, _ = _simulate_route(new_j, inst, coords, dist_mat)
                        if feasible_i and feasible_j:
                            old_cost = _route_cost(routes[i], inst, coords, dist_mat) + _route_cost(routes[j], inst, coords, dist_mat)
                            new_cost = _route_cost(new_i, inst, coords, dist_mat) + _route_cost(new_j, inst, coords, dist_mat)
                            if new_cost < old_cost - 1e-9:
                                routes[i] = new_i
                                routes[j] = new_j
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return routes


def plan(instance):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    coords = instance["coords"]
    n_nodes = len(coords)
    n_cust = instance["n_customers"]
    stations = instance["stations"]

    # Precompute distance matrix
    dist_mat = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = _euclid(coords[i], coords[j])
            dist_mat[i][j] = d
            dist_mat[j][i] = d

    # Initial construction
    routes = _greedy_construct(instance, coords, dist_mat)

    # Local search: intra-route improvements
    for _ in range(20):
        if time.time() > deadline:
            break
        changed = False
        for ri in range(len(routes)):
            if time.time() > deadline:
                break
            new_route = _two_opt(routes[ri], instance, coords, dist_mat)
            if new_route != routes[ri]:
                routes[ri] = new_route
                changed = True
            new_route = _or_opt(routes[ri], instance, coords, dist_mat)
            if new_route != routes[ri]:
                routes[ri] = new_route
                changed = True
        if not changed:
            break

    # Inter-route improvements
    for _ in range(10):
        if time.time() > deadline:
            break
        routes = _inter_route_moves(routes, instance, coords, dist_mat)

    # Final intra-route polish
    for _ in range(10):
        if time.time() > deadline:
            break
        changed = False
        for ri in range(len(routes)):
            if time.time() > deadline:
                break
            new_route = _two_opt(routes[ri], instance, coords, dist_mat)
            if new_route != routes[ri]:
                routes[ri] = new_route
                changed = True
        if not changed:
            break

    # Verify all customers are served exactly once
    served = []
    for route in routes:
        for node in route:
            if node not in stations:
                served.append(node)
    served.sort()
    expected = list(range(1, n_cust + 1))
    if served != expected:
        # Fallback: one customer per route
        routes = [[v] for v in range(1, n_cust + 1)]

    return routes
# EVOLVE-BLOCK-END
