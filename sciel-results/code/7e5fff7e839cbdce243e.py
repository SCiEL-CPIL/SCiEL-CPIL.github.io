"""Optimized EV routing with time windows and battery constraints."""
# EVOLVE-BLOCK-START
import numpy as np
from math import sqrt
import time
import random

DEADLINE_SECONDS = 110.0


def _build_distance_matrix(coords):
    n = len(coords)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = sqrt((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2)
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _simulate_route(route, instance, dist):
    """
    Simulate a route and return (total_distance, is_valid).
    route is a list of node ids (customers and/or stations), depot implicit at ends.
    """
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    station_set = instance["_station_set"]

    n_customers = instance["n_customers"]

    # Check capacity
    total_demand = 0
    for v in route:
        if v <= n_customers:
            total_demand += demands[v]
    if total_demand > capacity:
        return 0.0, False

    total_dist = 0.0
    battery = battery_max
    current_time = 0.0
    current_node = 0  # depot

    for v in route:
        d = dist[current_node, v]
        total_dist += d
        battery -= d
        if battery < -1e-9:
            return total_dist, False
        current_time += d / speed

        if v in station_set:
            recharge_time = (battery_max - battery) * recharge_rate
            current_time += recharge_time
            battery = battery_max
        else:
            # Customer
            if current_time < ready[v]:
                current_time = ready[v]
            if current_time > due[v] + 1e-9:
                return total_dist, False
            current_time += service[v]

        current_node = v

    # Return to depot
    d = dist[current_node, 0]
    total_dist += d
    battery -= d
    if battery < -1e-9:
        return total_dist, False
    current_time += d / speed

    if current_time > horizon + 1e-9:
        return total_dist, False

    return total_dist, True


def _validate_solution(routes, instance, dist):
    """Check if all customers are served exactly once and all routes are valid."""
    n_customers = instance["n_customers"]
    served = set()
    for route in routes:
        for v in route:
            if v <= n_customers:
                if v in served:
                    return False
                served.add(v)
            if v not in instance["_station_set"] and v > n_customers and v != 0:
                return False

    if served != set(range(1, n_customers + 1)):
        return False

    for route in routes:
        _, valid = _simulate_route(route, instance, dist)
        if not valid:
            return False
    return True


def _construct_clustered(instance, dist):
    """
    Construct initial solution using k-means-like clustering.
    """
    n_customers = instance["n_customers"]
    coords = np.array(instance["coords"])
    customers = list(range(1, n_customers + 1))
    customer_coords = coords[1:n_customers + 1]

    # Determine number of routes based on capacity
    demands = instance["demands"]
    total_demand = sum(demands[c] for c in customers)
    capacity = instance["capacity"]
    min_routes = max(1, int(np.ceil(total_demand / capacity)))

    # Use k-means clustering with k = min_routes (or a few more)
    k = min_routes + 2  # start with a few extra routes
    k = min(k, n_customers)

    # Simple k-means
    random.seed(42)
    centers = customer_coords[random.sample(range(n_customers), k)]
    assignments = [0] * n_customers

    for _ in range(20):  # iterations
        # Assign
        for i in range(n_customers):
            dists = np.linalg.norm(customer_coords - centers, axis=1)
            assignments[i] = np.argmin(dists)

        # Update centers
        for j in range(k):
            members = [customer_coords[i] for i in range(n_customers) if assignments[i] == j]
            if members:
                centers[j] = np.mean(members, axis=0)

    # Build routes from clusters
    routes = [[] for _ in range(k)]
    for i, c in enumerate(customers):
        routes[assignments[i - 1]].append(c)

    # Sort each route by due time, then apply 2-opt
    for i in range(len(routes)):
        if routes[i]:
            routes[i].sort(key=lambda v: instance["due"][v])

    # Remove empty routes
    routes = [r for r in routes if r]

    return routes


def _two_opt_route(route, instance, dist, max_iterations=30):
    """Apply 2-opt to a single route to minimize distance, maintaining validity."""
    n = len(route)
    if n < 2:
        return route

    best_route = route[:]
    best_dist, _ = _simulate_route(best_route, instance, dist)

    for _ in range(max_iterations):
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                d, valid = _simulate_route(new_route, instance, dist)
                if valid and d < best_dist - 1e-9:
                    best_route = new_route
                    best_dist = d
                    improved = True
                    route = new_route[:]
        if not improved:
            break

    return best_route


def _or_opt_route(route, instance, dist, max_iterations=15):
    """Apply OR-opt moves: relocate 1, 2, or 3 consecutive nodes to a new position."""
    n = len(route)
    if n < 3:
        return route

    best_route = route[:]
    best_dist, _ = _simulate_route(best_route, instance, dist)

    for _ in range(max_iterations):
        improved = False
        for length in [1, 2, 3]:
            for i in range(n - length + 1):
                segment = route[i:i + length]
                for j in range(n - length + 1):
                    if j >= i and j < i + length:
                        continue
                    # Remove segment
                    new_route = route[:i] + route[i + length:]
                    # Adjust j if j > i
                    insert_pos = j if j <= i else j - length
                    # Insert segment at new position
                    new_route = new_route[:insert_pos] + segment + new_route[insert_pos:]
                    if new_route == route:
                        continue
                    d, valid = _simulate_route(new_route, instance, dist)
                    if valid and d < best_dist - 1e-9:
                        best_route = new_route
                        best_dist = d
                        improved = True
                        route = new_route[:]
        if not improved:
            break

    return best_route


def _relocate_improve(route, instance, dist, max_iterations=15):
    """Apply relocate moves to a single route."""
    n = len(route)
    if n < 3:
        return route

    best_route = route[:]
    best_dist, _ = _simulate_route(best_route, instance, dist)

    for _ in range(max_iterations):
        improved = False
        for i in range(n):
            for j in range(n):
                if i == j or abs(i - j) == 1:
                    continue
                v = route[i]
                new_route = route[:i] + route[i + 1:]
                insert_pos = j if j < i else j
                new_route = new_route[:insert_pos] + [v] + new_route[insert_pos:]
                d, valid = _simulate_route(new_route, instance, dist)
                if valid and d < best_dist - 1e-9:
                    best_route = new_route
                    best_dist = d
                    improved = True
                    route = new_route[:]
        if not improved:
            break

    return best_route


def _swap_improve(route, instance, dist, max_iterations=15):
    """Apply swap moves to a single route."""
    n = len(route)
    if n < 2:
        return route

    best_route = route[:]
    best_dist, _ = _simulate_route(best_route, instance, dist)

    for _ in range(max_iterations):
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                new_route = route[:]
                new_route[i], new_route[j] = new_route[j], new_route[i]
                d, valid = _simulate_route(new_route, instance, dist)
                if valid and d < best_dist - 1e-9:
                    best_route = new_route
                    best_dist = d
                    improved = True
                    route = new_route[:]
        if not improved:
            break

    return best_route


def _try_merge_routes(route1, route2, instance, dist):
    """
    Try to merge two routes into one.
    Return the merged route if valid and shorter, else None.
    """
    # Try both orderings
    for merged in [route1 + route2, route2 + route1]:
        d, valid = _simulate_route(merged, instance, dist)
        if valid:
            d1, _ = _simulate_route(route1, instance, dist)
            d2, _ = _simulate_route(route2, instance, dist)
            if d < d1 + d2 - 1e-9:
                return merged
    return None


def _cross_move(route1, route2, instance, dist):
    """Try a cross exchange between two routes."""
    n1 = len(route1)
    n2 = len(route2)
    if n1 < 2 or n2 < 2:
        return None

    d1, _ = _simulate_route(route1, instance, dist)
    d2, _ = _simulate_route(route2, instance, dist)
    current_total = d1 + d2

    for i in range(1, n1):
        for j in range(1, n2):
            # Swap suffixes
            new_route1 = route1[:i] + route2[j:]
            new_route2 = route2[:j] + route1[i:]

            d1_new, v1 = _simulate_route(new_route1, instance, dist)
            if not v1:
                continue
            d2_new, v2 = _simulate_route(new_route2, instance, dist)
            if not v2:
                continue

            if d1_new + d2_new < current_total - 1e-9:
                return (new_route1, new_route2, d1_new + d2_new)

    return None


def _simulated_annealing(routes, instance, dist, start_time, deadline):
    """
    Apply simulated annealing to improve the solution.
    """
    n_customers = instance["n_customers"]
    best_routes = routes[:]
    best_total_dist = sum(_simulate_route(r, instance, dist)[0] for r in routes)

    T = 100.0
    T_min = 1.0
    cooling_rate = 0.995
    max_iter = 5000

    current_routes = routes[:]
    current_total_dist = best_total_dist

    for it in range(max_iter):
        if time.time() > deadline:
            break

        # Choose a random move
        move_type = random.random()
        if move_type < 0.3:
            # 2-opt within a route
            ri = random.randint(0, len(current_routes) - 1)
            route = current_routes[ri]
            n = len(route)
            if n < 2:
                continue
            i, j = sorted(random.sample(range(n), 2))
            new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
            d, valid = _simulate_route(new_route, instance, dist)
            if not valid:
                continue
            old_d, _ = _simulate_route(route, instance, dist)
            delta = d - old_d
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_routes[ri] = new_route
                current_total_dist += delta
        elif move_type < 0.5:
            # Relocate a customer to another position in the same route
            ri = random.randint(0, len(current_routes) - 1)
            route = current_routes[ri]
            n = len(route)
            if n < 3:
                continue
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j or abs(i - j) == 1:
                continue
            v = route[i]
            new_route = route[:i] + route[i + 1:]
            insert_pos = j if j < i else j
            new_route = new_route[:insert_pos] + [v] + new_route[insert_pos:]
            d, valid = _simulate_route(new_route, instance, dist)
            if not valid:
                continue
            old_d, _ = _simulate_route(route, instance, dist)
            delta = d - old_d
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_routes[ri] = new_route
                current_total_dist += delta
        elif move_type < 0.7:
            # Swap two customers in the same route
            ri = random.randint(0, len(current_routes) - 1)
            route = current_routes[ri]
            n = len(route)
            if n < 2:
                continue
            i, j = sorted(random.sample(range(n), 2))
            new_route = route[:]
            new_route[i], new_route[j] = new_route[j], new_route[i]
            d, valid = _simulate_route(new_route, instance, dist)
            if not valid:
                continue
            old_d, _ = _simulate_route(route, instance, dist)
            delta = d - old_d
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_routes[ri] = new_route
                current_total_dist += delta
        else:
            # Relocate a customer to another route
            if len(current_routes) < 2:
                continue
            ri = random.randint(0, len(current_routes) - 1)
            route = current_routes[ri]
            n = len(route)
            if n < 2:
                continue
            i = random.randint(0, n - 1)
            v = route[i]
            new_route1 = route[:i] + route[i + 1:]
            rj = random.randint(0, len(current_routes) - 1)
            if rj == ri:
                continue
            route2 = current_routes[rj]
            j = random.randint(0, len(route2))
            new_route2 = route2[:j] + [v] + route2[j:]

            d1, v1 = _simulate_route(new_route1, instance, dist)
            if not v1:
                continue
            d2, v2 = _simulate_route(new_route2, instance, dist)
            if not v2:
                continue
            old_d1, _ = _simulate_route(route, instance, dist)
            old_d2, _ = _simulate_route(route2, instance, dist)
            delta = (d1 + d2) - (old_d1 + old_d2)
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_routes[ri] = new_route1
                current_routes[rj] = new_route2
                current_total_dist += delta
                # Remove empty routes
                current_routes = [r for r in current_routes if r]

        # Update best
        if current_total_dist < best_total_dist:
            best_total_dist = current_total_dist
            best_routes = [r[:] for r in current_routes]

        T *= cooling_rate
        if T < T_min:
            T = T_min

    return best_routes


def plan(instance):
    start_time = time.time()

    # Precompute distance matrix
    coords = instance["coords"]
    dist = _build_distance_matrix(coords)

    # Precompute station set for faster lookup
    instance["_station_set"] = set(instance["stations"])

    # Initial solution using clustered construction
    routes = _construct_clustered(instance, dist)

    # Initial local search
    for i in range(len(routes)):
        routes[i] = _two_opt_route(routes[i], instance, dist, max_iterations=20)
        routes[i] = _or_opt_route(routes[i], instance, dist, max_iterations=10)
        routes[i] = _relocate_improve(routes[i], instance, dist, max_iterations=10)
        routes[i] = _swap_improve(routes[i], instance, dist, max_iterations=10)

    # Try to merge routes
    merged = True
    while merged and time.time() < start_time + DEADLINE_SECONDS - 15:
        merged = False
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                merged_route = _try_merge_routes(routes[i], routes[j], instance, dist)
                if merged_route is not None:
                    routes[i] = merged_route
                    routes.pop(j)
                    merged = True
                    # Re-optimize the changed route
                    routes[i] = _two_opt_route(routes[i], instance, dist, max_iterations=15)
                    routes[i] = _or_opt_route(routes[i], instance, dist, max_iterations=10)
                    break
            if merged:
                break

    # Simulated annealing
    if time.time() < start_time + DEADLINE_SECONDS - 5:
        routes = _simulated_annealing(routes, instance, dist, start_time, start_time + DEADLINE_SECONDS - 5)

    # Final local search
    for i in range(len(routes)):
        routes[i] = _two_opt_route(routes[i], instance, dist, max_iterations=10)
        routes[i] = _or_opt_route(routes[i], instance, dist, max_iterations=8)
        routes[i] = _relocate_improve(routes[i], instance, dist, max_iterations=8)
        routes[i] = _swap_improve(routes[i], instance, dist, max_iterations=8)

    # Inter-route cross moves
    if time.time() < start_time + DEADLINE_SECONDS - 2:
        improved = True
        while improved and time.time() < start_time + DEADLINE_SECONDS - 2:
            improved = False
            for i in range(len(routes)):
                for j in range(i + 1, len(routes)):
                    result = _cross_move(routes[i], routes[j], instance, dist)
                    if result is not None:
                        routes[i], routes[j], _ = result
                        improved = True
                        # Re-optimize the changed routes
                        routes[i] = _two_opt_route(routes[i], instance, dist, max_iterations=8)
                        routes[j] = _two_opt_route(routes[j], instance, dist, max_iterations=8)

    # Final validation
    if not _validate_solution(routes, instance, dist):
        # Fallback to one customer per route
        n_customers = instance["n_customers"]
        routes = [[v] for v in range(1, n_customers + 1)]

    return routes
# EVOLVE-BLOCK-END
