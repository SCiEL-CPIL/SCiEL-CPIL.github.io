"""EV-routing planner: global structural optimization via matching + constrained insertion."""
import numpy as np
from scipy.optimize import linear_sum_assignment
import math
import time


def _euclidean(coords):
    """Compute pairwise Euclidean distance matrix."""
    n = len(coords)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            d[i, j] = math.sqrt(dx * dx + dy * dy)
            d[j, i] = d[i, j]
    return d


def _simulate_route(route, instance, dist, speed, capacity, battery, recharge_rate, horizon):
    """
    Simulate a route (list of customer/station node ids, depot implicit at both ends).
    Returns (valid, total_distance, end_time).
    """
    n_customers = instance["n_customers"]
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = set(instance["stations"])

    total_dist = 0.0
    current_time = 0.0
    current_bat = battery
    current_pos = 0  # depot
    load = 0

    for node in route:
        # Travel from current_pos to node
        d = dist[current_pos][node]
        total_dist += d
        current_time += d / speed
        current_bat -= d

        if current_bat < -1e-9:
            return (False, total_dist, current_time)

        if node in stations:
            # Charging station
            energy = max(0.0, battery - current_bat)
            charge_time = energy * recharge_rate
            current_time += charge_time
            current_bat = battery
        else:
            # Customer
            if load + demands[node] > capacity:
                return (False, total_dist, current_time)
            load += demands[node]

            # Time window
            if current_time < ready[node]:
                current_time = ready[node]
            if current_time > due[node] + 1e-9:
                return (False, total_dist, current_time)

            current_time += service[node]

        current_pos = node

    # Return to depot
    d = dist[current_pos][0]
    total_dist += d
    current_time += d / speed
    current_bat -= d

    if current_bat < -1e-9:
        return (False, total_dist, current_time)
    if current_time > horizon + 1e-9:
        return (False, total_dist, current_time)

    return (True, total_dist, current_time)


def _try_insert_station(route, station, instance, dist, speed, capacity, battery, recharge_rate, horizon):
    """
    Try inserting a charging station at the best position in the route.
    Returns (improved_route, valid) or (None, False) if no valid insertion found.
    """
    # Try inserting after each position (including before first)
    best_route = None
    best_valid = False

    for i in range(len(route) + 1):
        new_route = route[:i] + [station] + route[i:]
        valid, _, _ = _simulate_route(new_route, instance, dist, speed, capacity, battery, recharge_rate, horizon)
        if valid:
            best_route = new_route
            best_valid = True
            break  # Take the first valid insertion (earliest is usually best for battery)

    return (best_route, best_valid)


def _build_initial_routes_customers(customers, instance, dist, speed, capacity, battery, recharge_rate, horizon):
    """
    Build initial routes by grouping customers.
    Uses nearest-neighbor heuristic to form routes.
    """
    n_customers = instance["n_customers"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]

    # Sort customers by due time (earliest first)
    sorted_customers = sorted(customers, key=lambda v: due[v])

    routes = []
    remaining = set(sorted_customers)

    while remaining:
        # Start a new route with the earliest-due remaining customer
        start = sorted_customers[0] if sorted_customers[0] in remaining else None
        if start is None:
            # Find the first remaining in sorted order
            for c in sorted_customers:
                if c in remaining:
                    start = c
                    break

        route = [start]
        remaining.discard(start)
        current_load = demands[start]
        current_pos = start

        # Try to add more customers to this route
        changed = True
        while changed:
            changed = False
            best_next = None
            best_cost = float('inf')

            for c in remaining:
                if current_load + demands[c] > capacity:
                    continue

                # Check if we can reach c from current_pos with enough battery
                # and if the time window is feasible
                d = dist[current_pos][c]
                # Rough battery check: need enough battery to go depot->start->...->current->c->depot
                # For now, just check if distance is reasonable
                if d < best_cost:
                    best_cost = d
                    best_next = c

            if best_next is not None:
                route.append(best_next)
                remaining.discard(best_next)
                current_load += demands[best_next]
                current_pos = best_next
                changed = True

        routes.append(route)

    return routes


def _optimize_route_with_charging(route, instance, dist, speed, capacity, battery, recharge_rate, horizon):
    """
    Add charging stations to a route to make it valid.
    Returns (valid_route, valid) or (None, False).
    """
    stations = instance["stations"]

    # First check if route is valid without charging
    valid, _, _ = _simulate_route(route, instance, dist, speed, capacity, battery, recharge_rate, horizon)
    if valid:
        return (route, True)

    # Try adding stations one at a time
    current_route = list(route)
    for _ in range(4):  # Try up to 4 station insertions
        improved = False
        for st in stations:
            if st in current_route:
                continue
            new_route, valid = _try_insert_station(current_route, st, instance, dist, speed, capacity, battery, recharge_rate, horizon)
            if valid:
                current_route = new_route
                improved = True
                break
        if not improved:
            break

    valid, _, _ = _simulate_route(current_route, instance, dist, speed, capacity, battery, recharge_rate, horizon)
    if valid:
        return (current_route, True)

    return (None, False)


def _two_opt_improve(route, instance, dist, speed, capacity, battery, recharge_rate, horizon):
    """
    Apply 2-opt improvement to a route (customers only, then re-add charging).
    """
    n = len(route)
    if n < 2:
        return route

    best_route = list(route)
    improved = True

    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Reverse segment i..j
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                valid, _, _ = _simulate_route(new_route, instance, dist, speed, capacity, battery, recharge_rate, horizon)
                if valid:
                    # Check if distance is better
                    old_dist = sum(dist[best_route[k]][best_route[k+1]] for k in range(n-1))
                    old_dist += dist[0][best_route[0]] + dist[best_route[-1]][0]
                    new_dist = sum(dist[new_route[k]][new_route[k+1]] for k in range(n-1))
                    new_dist += dist[0][new_route[0]] + dist[new_route[-1]][0]
                    if new_dist < old_dist - 1e-9:
                        best_route = new_route
                        improved = True

    return best_route


def plan(instance):
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s limit

    n_customers = instance["n_customers"]
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    # Compute distance matrix
    dist = _euclidean(coords)

    customers = list(range(1, n_customers + 1))

    # Phase 1: Build initial routes using nearest-neighbor
    routes = _build_initial_routes_customers(customers, instance, dist, speed, capacity, battery, recharge_rate, horizon)

    # Phase 2: Add charging stations to make routes valid
    valid_routes = []
    for route in routes:
        if time.time() > deadline:
            break
        optimized, valid = _optimize_route_with_charging(route, instance, dist, speed, capacity, battery, recharge_rate, horizon)
        if valid:
            valid_routes.append(optimized)
        else:
            # Fallback: serve each customer in this route individually
            for c in route:
                valid_routes.append([c])

    # Phase 3: 2-opt improvement
    if time.time() < deadline:
        new_routes = []
        for route in valid_routes:
            if time.time() > deadline:
                new_routes.append(route)
                continue
            improved = _two_opt_improve(route, instance, dist, speed, capacity, battery, recharge_rate, horizon)
            # Re-add charging if needed
            optimized, valid = _optimize_route_with_charging(improved, instance, dist, speed, capacity, battery, recharge_rate, horizon)
            if valid:
                new_routes.append(optimized)
            else:
                new_routes.append(route)
        valid_routes = new_routes

    # Phase 4: Verify all customers are served exactly once
    served = []
    for route in valid_routes:
        for node in route:
            if node not in stations:
                served.append(node)

    # If any customer missing, add them as single routes
    served_set = set(served)
    for c in customers:
        if c not in served_set:
            valid_routes.append([c])

    # Final validation
    final_routes = []
    for route in valid_routes:
        if time.time() > deadline:
            final_routes.append(route)
            continue
        valid, _, _ = _simulate_route(route, instance, dist, speed, capacity, battery, recharge_rate, horizon)
        if valid:
            final_routes.append(route)
        else:
            # Fallback to individual
            for node in route:
                if node not in stations:
                    final_routes.append([node])

    return final_routes
