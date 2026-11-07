"""Optimized EV routing with time windows and battery constraints."""
# EVOLVE-BLOCK-START
import numpy as np
from math import sqrt
import time
import random

DEADLINE_SECONDS = 110.0


def _euclidean(p1, p2):
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _validate_route(route, instance):
    coords = np.array(instance["coords"], dtype=np.float64)
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])

    n = len(coords)
    # Check customer visit validity
    total_demand = 0
    for node in route:
        if 1 <= node <= instance["n_customers"]:
            total_demand += demands[node]
    if total_demand > capacity:
        return False, 0.0

    # Simulate route
    cur = 0  # depot
    t = 0.0
    battery = battery_max
    dist = 0.0

    for node in route:
        d = _euclidean(coords[cur], coords[node])
        battery -= d
        t += d / speed
        if battery < 0:
            return False, 0.0
        dist += d
        cur = node

        if node in stations:
            # Recharge
            t += (battery_max - battery) * recharge_rate
            battery = battery_max
        else:
            # Customer
            t = max(t, ready[node])
            if t > due[node]:
                return False, 0.0
            t += service[node]

    # Return to depot
    d = _euclidean(coords[cur], coords[0])
    battery -= d
    t += d / speed
    dist += d
    if battery < 0:
        return False, 0.0
    if t > horizon:
        return False, 0.0

    return True, dist


def _validate_solution(routes, instance):
    # Check all customers served exactly once
    all_visited = []
    for route in routes:
        for node in route:
            if 1 <= node <= instance["n_customers"]:
                all_visited.append(node)
    if len(all_visited) != instance["n_customers"]:
        return False, 0.0
    if len(set(all_visited)) != instance["n_customers"]:
        return False, 0.0

    total_dist = 0.0
    for route in routes:
        valid, d = _validate_route(route, instance)
        if not valid:
            return False, 0.0
        total_dist += d
    return True, total_dist


def _compute_total_distance(routes, instance):
    coords = np.array(instance["coords"], dtype=np.float64)
    total = 0.0
    for route in routes:
        cur = 0
        for node in route:
            total += _euclidean(coords[cur], coords[node])
            cur = node
        total += _euclidean(coords[cur], coords[0])
    return total


def _construct_nearest_neighbor(remaining, route, instance, coords, demands, ready, due, service, capacity, battery_max, speed, recharge_rate, stations_set):
    """Extend a route using nearest neighbor heuristic with feasibility check."""
    if not remaining:
        return route, remaining, True

    cur = 0 if not route else route[-1]
    t = 0.0
    battery = battery_max
    total_demand = sum(demands[v] for v in route if 1 <= v <= instance["n_customers"])

    # Compute current time and battery after the existing route
    if route:
        cur_node = 0
        for node in route:
            d = _euclidean(coords[cur_node], coords[node])
            battery -= d
            t += d / speed
            cur_node = node
            if node in stations_set:
                t += (battery_max - battery) * recharge_rate
                battery = battery_max
            else:
                t = max(t, ready[node])
                t += service[node]
        cur = cur_node

    best_node = None
    best_dist = float('inf')

    for cand in remaining:
        if 1 <= cand <= instance["n_customers"]:
            if total_demand + demands[cand] > capacity:
                continue
        d = _euclidean(coords[cur], coords[cand])
        new_battery = battery - d
        new_t = t + d / speed
        if new_battery < 0:
            continue
        if cand in stations_set:
            new_t += (battery_max - new_battery) * recharge_rate
        else:
            new_t = max(new_t, ready[cand])
            if new_t > due[cand]:
                continue

        if d < best_dist:
            best_dist = d
            best_node = cand

    if best_node is not None:
        route.append(best_node)
        remaining.remove(best_node)
        return route, remaining, True
    return route, remaining, False


def plan(instance):
    start_time = time.time()
    coords = np.array(instance["coords"], dtype=np.float64)
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations_set = set(instance["stations"])
    n_customers = instance["n_customers"]

    customers = list(range(1, n_customers + 1))

    # Sort by due date for initial construction
    customers_sorted = sorted(customers, key=lambda v: due[v])

    # Build routes using savings/nearest neighbor approach
    # Start with each customer as its own route, then merge
    # For simplicity, use a constructive heuristic: build routes greedily

    def build_routes_greedy():
        remaining = set(customers_sorted)
        routes = []
        # Process customers in order of due date, adding to feasible routes or creating new
        for cust in customers_sorted:
            added = False
            for r in routes:
                # Try to insert cust into route r
                # Simple: append to end
                if _try_append(r, cust, instance, coords, demands, ready, due, service,
                               capacity, battery_max, speed, recharge_rate, stations_set):
                    r.append(cust)
                    added = True
                    break
            if not added:
                routes.append([cust])
        return routes

    def _try_append(route, node, instance, coords, demands, ready, due, service,
                    capacity, battery_max, speed, recharge_rate, stations_set):
        """Check if node can be appended to route."""
        total_demand = sum(demands[v] for v in route if 1 <= v <= instance["n_customers"])
        if 1 <= node <= instance["n_customers"]:
            if total_demand + demands[node] > capacity:
                return False

        # Simulate route + node
        cur = 0
        t = 0.0
        battery = battery_max
        for n in route + [node]:
            d = _euclidean(coords[cur], coords[n])
            battery -= d
            t += d / speed
            if battery < 0:
                return False
            cur = n
            if n in stations_set:
                t += (battery_max - battery) * recharge_rate
                battery = battery_max
            else:
                t = max(t, ready[n])
                if t > due[n]:
                    return False
                t += service[n]

        # Return to depot
        d = _euclidean(coords[cur], coords[0])
        battery -= d
        t += d / speed
        if battery < 0:
            return False
        if t > horizon:
            return False
        return True

    def optimize_route(route, instance, coords, demands, ready, due, service,
                       capacity, battery_max, speed, recharge_rate, stations_set):
        """Try 2-opt style improvement on a route."""
        if len(route) < 2:
            return route

        best_route = list(route)
        best_dist = _compute_total_distance([best_route], instance)
        improved = True
        max_iters = 20
        iters = 0

        while improved and iters < max_iters:
            improved = False
            iters += 1
            for i in range(len(best_route) - 1):
                for j in range(i + 1, len(best_route)):
                    candidate = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                    if _validate_route(candidate, instance)[0]:
                        d = _compute_total_distance([candidate], instance)
                        if d < best_dist - 1e-9:
                            best_route = candidate
                            best_dist = d
                            improved = True
        return best_route

    def merge_routes(routes, instance, coords, demands, ready, due, service,
                     capacity, battery_max, speed, recharge_rate, stations_set):
        """Try to merge two routes to reduce total distance."""
        if len(routes) < 2:
            return routes

        merged = True
        while merged:
            merged = False
            best_merge = None
            best_saving = 0
            for i in range(len(routes)):
                for j in range(i + 1, len(routes)):
                    # Try concatenating route i + route j
                    combined = routes[i] + routes[j]
                    if _validate_route(combined, instance)[0]:
                        old_dist = _compute_total_distance([routes[i]], instance) + _compute_total_distance([routes[j]], instance)
                        new_dist = _compute_total_distance([combined], instance)
                        saving = old_dist - new_dist
                        if saving > best_saving + 1e-9:
                            best_saving = saving
                            best_merge = (i, j, combined)

            if best_merge:
                i, j, combined = best_merge
                # Remove i and j (j comes after i, so remove j first if j > i)
                routes.pop(j)
                routes.pop(i)
                routes.append(combined)
                merged = True

        return routes

    # Initial construction
    routes = build_routes_greedy()

    # Validate
    valid, total_dist = _validate_solution(routes, instance)
    if not valid:
        # Fallback: one customer per route
        return [[v] for v in customers_sorted]

    # Local optimization
    deadline_check = 5.0
    for r in routes:
        if time.time() - start_time > deadline_check:
            break
        routes[routes.index(r)] = optimize_route(r, instance, coords, demands, ready, due, service,
                                                  capacity, battery_max, speed, recharge_rate, stations_set)

    # Merge routes
    if time.time() - start_time < deadline_check + 5.0:
        routes = merge_routes(routes, instance, coords, demands, ready, due, service,
                              capacity, battery_max, speed, recharge_rate, stations_set)

    # Final validation
    valid, total_dist = _validate_solution(routes, instance)
    if not valid:
        return [[v] for v in customers_sorted]

    return routes
# EVOLVE-BLOCK-END
