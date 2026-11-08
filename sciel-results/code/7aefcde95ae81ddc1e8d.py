"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random

DEADLINE_MARGIN = 5.0  # seconds to keep before the 120s limit


def _euclidean(coords, i, j):
    dx = coords[i, 0] - coords[j, 0]
    dy = coords[i, 1] - coords[j, 1]
    return math.sqrt(dx * dx + dy * dy)


def _build_distance_matrix(coords):
    n = len(coords)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            d[i, j] = d[j, i] = math.sqrt(dx * dx + dy * dy)
    return d


def _simulate(route, dist_mat, demands, ready, due, service, stations,
              capacity, max_battery, speed, recharge_rate, horizon, coords):
    """
    Simulate a route. Returns (is_valid, total_distance).
    route: list of node ids (customers and/or stations), depot implicit at start/end.
    """
    if not route:
        return True, 0.0

    cur = 0
    t = 0.0
    bat = max_battery
    total_dist = 0.0
    load = 0.0

    for node in route:
        d = dist_mat[cur, node]
        t += d / speed
        bat -= d
        total_dist += d

        if bat < -1e-9:
            return False, total_dist

        if node in stations:
            # Charging station: recharge time = (max_battery - bat) * recharge_rate
            charge_time = (max_battery - bat) * recharge_rate
            t += charge_time
            bat = max_battery
        else:
            # Customer
            load += demands[node]
            if load > capacity + 1e-9:
                return False, total_dist
            if t < ready[node]:
                t = ready[node]
            if t > due[node] + 1e-9:
                return False, total_dist
            t += service[node]

    # Return to depot
    d = dist_mat[cur, 0]
    t += d / speed
    bat -= d
    total_dist += d

    if bat < -1e-9:
        return False, total_dist
    if t > horizon + 1e-9:
        return False, total_dist

    return True, total_dist


def _solve_instance(instance, deadline):
    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    max_battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    n_nodes = len(coords)
    dist_mat = _build_distance_matrix(coords)

    customers = list(range(1, n_customers + 1))

    # Precompute distances from depot to all nodes
    depot_to = dist_mat[0]

    # Greedy construction: nearest-neighbor with time-window feasibility
    # Sort customers by due time as a heuristic for ordering
    customers_sorted = sorted(customers, key=lambda c: due[c])

    # Build initial routes using a greedy insertion approach
    # Start with no routes, insert customers one by one
    routes = []  # list of lists of node ids

    # For each customer, try to insert into an existing route or create a new one
    for cust in customers_sorted:
        best_route_idx = -1
        best_pos = -1
        best_dist = float('inf')

        for r_idx, route in enumerate(routes):
            # Try inserting cust at each position
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [cust] + route[pos:]
                valid, dist = _simulate(
                    new_route, dist_mat, demands, ready, due, service,
                    stations, capacity, max_battery, speed, recharge_rate, horizon, coords
                )
                if valid and dist < best_dist:
                    best_dist = dist
                    best_route_idx = r_idx
                    best_pos = pos

        if best_route_idx >= 0:
            routes[best_route_idx] = routes[best_route_idx][:best_pos] + [cust] + routes[best_route_idx][best_pos:]
        else:
            # Create a new route with just this customer
            # Check if it's feasible as a standalone route
            valid, dist = _simulate(
                [cust], dist_mat, demands, ready, due, service,
                stations, capacity, max_battery, speed, recharge_rate, horizon, coords
            )
            if valid:
                routes.append([cust])
            else:
                # Try adding a station before the customer
                for st in stations:
                    valid2, dist2 = _simulate(
                        [st, cust], dist_mat, demands, ready, due, service,
                        stations, capacity, max_battery, speed, recharge_rate, horizon, coords
                    )
                    if valid2:
                        routes.append([st, cust])
                        break
                else:
                    # Fallback: single customer (may be invalid but we need to assign)
                    routes.append([cust])

    # Local search: 2-opt and relocate improvements
    start_time = time.time()
    improved = True
    max_iters = 500

    while improved and (time.time() - start_time) < (deadline - DEADLINE_MARGIN):
        improved = False
        for r_idx in range(len(routes)):
            route = routes[r_idx]
            # 2-opt on customer positions (ignoring stations for simplicity, or including all)
            n = len(route)
            for i in range(n):
                for j in range(i + 1, n):
                    new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                    valid, dist_new = _simulate(
                        new_route, dist_mat, demands, ready, due, service,
                        stations, capacity, max_battery, speed, recharge_rate, horizon, coords
                    )
                    valid_old, dist_old = _simulate(
                        route, dist_mat, demands, ready, due, service,
                        stations, capacity, max_battery, speed, recharge_rate, horizon, coords
                    )
                    if valid and valid_old and dist_new < dist_old - 1e-6:
                        routes[r_idx] = new_route
                        improved = True

            # Relocate: move a customer from this route to another
            for c_idx, cust in enumerate(route):
                for r2_idx in range(len(routes)):
                    if r2_idx == r_idx:
                        continue
                    route2 = routes[r2_idx]
                    for pos in range(len(route2) + 1):
                        # Remove cust from route
                        new_route1 = route[:c_idx] + route[c_idx + 1:]
                        # Insert into route2
                        new_route2 = route2[:pos] + [cust] + route2[pos:]
                        valid1, dist1 = _simulate(
                            new_route1, dist_mat, demands, ready, due, service,
                            stations, capacity, max_battery, speed, recharge_rate, horizon, coords
                        )
                        valid2, dist2 = _simulate(
                            new_route2, dist_mat, demands, ready, due, service,
                            stations, capacity, max_battery, speed, recharge_rate, horizon, coords
                        )
                        if valid1 and valid2:
                            old_dist = dist1 + dist2
                            # Compare with current total
                            _, cur_d1 = _simulate(route, dist_mat, demands, ready, due, service,
                                                   stations, capacity, max_battery, speed, recharge_rate, horizon, coords)
                            _, cur_d2 = _simulate(route2, dist_mat, demands, ready, due, service,
                                                  stations, capacity, max_battery, speed, recharge_rate, horizon, coords)
                            if old_dist < cur_d1 + cur_d2 - 1e-6:
                                routes[r_idx] = new_route1
                                routes[r2_idx] = new_route2
                                improved = True
                                break
                    if improved:
                        break

    # Remove empty routes
    routes = [r for r in routes if r]

    # Final validation
    total_dist = 0.0
    for route in routes:
        valid, d = _simulate(
            route, dist_mat, demands, ready, due, service,
            stations, capacity, max_battery, speed, recharge_rate, horizon, coords
        )
        if not valid:
            # Fallback: one customer per route
            fallback = [[c] for c in customers]
            return fallback
        total_dist += d

    return routes


def plan(instance):
    start_time = time.time()
    deadline = start_time + 115.0  # 115s out of 120s

    # Three fixed instances are expected; but the function receives one instance at a time
    # Actually, re-reading the task: "returns routes for 3 fixed generated instances"
    # But the interface says instance is a dict. Let me re-read...
    # "Write plan(instance), which returns routes for 3 fixed generated instances"
    # This is ambiguous. Looking at the seed program, it takes one instance and returns routes for it.
    # The seed returns [[v] for v in customers] which is a list of routes for one instance.
    # So plan(instance) takes one instance dict and returns a list of routes.
    # The "3 fixed generated instances" likely means the evaluator calls plan 3 times with 3 different instances.
    # So we just need to handle one instance per call.

    result = _solve_instance(instance, deadline)
    return result
# EVOLVE-BLOCK-END
