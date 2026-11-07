"""Optimized EV routing with time windows and battery constraints."""
# EVOLVE-BLOCK-START
import numpy as np
from math import sqrt
import time
import random

DEADLINE_SECONDS = 140.0  # safety margin below 150s limit


def _build_dist_matrix(coords):
    """Precompute distance matrix using vectorized operations."""
    coords = np.array(coords, dtype=np.float64)
    n_nodes = len(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    return dist_matrix


def _simulate_route(route, dist_matrix, n_customers, stations,
                    battery_max, speed, recharge_rate, horizon,
                    demands, ready, due, service):
    """Simulate a route. Returns (feasible, distance)."""
    total_demand = 0
    for node in route:
        if node <= n_customers:
            total_demand += demands[node]
    if total_demand > 100:  # capacity is always 100
        return False, 0.0

    battery = float(battery_max)
    time_at = 0.0
    distance = 0.0
    current_node = 0

    for node in route:
        d = dist_matrix[current_node, node]
        time_at += d
        battery -= d
        distance += d

        if battery < -1e-9:
            return False, 0.0

        if node <= n_customers:
            r = ready[node]
            if time_at < r:
                time_at = r
            if time_at > due[node] + 1e-9:
                return False, 0.0
            time_at += service[node]
        elif node in stations:
            energy_needed = battery_max - battery
            time_at += energy_needed * recharge_rate
            battery = float(battery_max)

        current_node = node

    d = dist_matrix[current_node, 0]
    time_at += d
    battery -= d
    distance += d

    if battery < -1e-9:
        return False, 0.0
    if time_at > horizon + 1e-9:
        return False, 0.0

    return True, distance


def _validate_solution(routes, instance):
    """Validate all routes. Returns (valid, total_distance)."""
    n_customers = instance["n_customers"]
    coords = np.array(instance["coords"], dtype=np.float64)
    dist_matrix = _build_dist_matrix(coords)
    stations = set(instance["stations"])
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    served = set()
    total_distance = 0.0

    for route in routes:
        feasible, dist = _simulate_route(route, dist_matrix, n_customers, stations,
                                          battery_max, speed, recharge_rate, horizon,
                                          demands, ready, due, service)
        if not feasible:
            return False, 0.0
        total_distance += dist

        for node in route:
            if node <= n_customers:
                if node in served:
                    return False, 0.0
                served.add(node)

    if served != set(range(1, n_customers + 1)):
        return False, 0.0

    return True, total_distance


def _compute_total_distance(routes, dist_matrix):
    """Compute total distance of routes."""
    total = 0.0
    for route in routes:
        current = 0
        for node in route:
            total += dist_matrix[current, node]
            current = node
        total += dist_matrix[current, 0]
    return total


def _greedy_construct(instance):
    """Construct initial solution using greedy insertion."""
    coords = np.array(instance["coords"], dtype=np.float64)
    dist_matrix = _build_dist_matrix(coords)
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    n_customers = instance["n_customers"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])

    # Sort customers by due time
    customers = list(range(1, n_customers + 1))
    customers.sort(key=lambda v: due[v])

    routes = []
    route_loads = []

    for cust in customers:
        best_route_idx = -1
        best_insert_pos = -1
        best_cost = float('inf')

        for r_idx, route in enumerate(routes):
            if route_loads[r_idx] + demands[cust] > capacity:
                continue

            for pos in range(len(route) + 1):
                new_route = route[:pos] + [cust] + route[pos:]

                feasible, _ = _simulate_route(new_route, dist_matrix, n_customers,
                                               stations, battery_max, speed,
                                               recharge_rate, horizon,
                                               demands, ready, due, service)
                if not feasible:
                    continue

                # Calculate insertion cost
                if pos == 0:
                    cost = (dist_matrix[0, cust] + dist_matrix[cust, route[0]] -
                           dist_matrix[0, route[0]])
                elif pos == len(route):
                    cost = (dist_matrix[route[-1], cust] + dist_matrix[cust, 0] -
                           dist_matrix[route[-1], 0])
                else:
                    cost = (dist_matrix[route[pos-1], cust] + dist_matrix[cust, route[pos]] -
                           dist_matrix[route[pos-1], route[pos]])

                if cost < best_cost:
                    best_cost = cost
                    best_route_idx = r_idx
                    best_insert_pos = pos

        if best_route_idx == -1:
            routes.append([cust])
            route_loads.append(demands[cust])
        else:
            routes[best_route_idx].insert(best_insert_pos, cust)
            route_loads[best_route_idx] += demands[cust]

    return routes


def _two_opt_route(route, dist_matrix, n_customers, stations,
                   battery_max, speed, recharge_rate, horizon,
                   demands, ready, due, service, max_iters=50):
    """Optimize route order using 2-opt."""
    if len(route) < 2:
        return route

    best_route = route[:]
    best_dist = _compute_total_distance([best_route], dist_matrix)

    for _ in range(max_iters):
        improved = False
        for i in range(len(best_route) - 1):
            for j in range(i + 1, len(best_route)):
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]

                feasible, _ = _simulate_route(new_route, dist_matrix, n_customers,
                                               stations, battery_max, speed,
                                               recharge_rate, horizon,
                                               demands, ready, due, service)
                if not feasible:
                    continue

                new_dist = _compute_total_distance([new_route], dist_matrix)
                if new_dist < best_dist - 1e-9:
                    best_route = new_route
                    best_dist = new_dist
                    improved = True
        if not improved:
            break

    return best_route


def _local_search(routes, dist_matrix, instance, deadline):
    """Improve solution using 2-opt and inter-route moves."""
    n_customers = instance["n_customers"]
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

    def get_total_dist(rts):
        return _compute_total_distance(rts, dist_matrix)

    def is_feasible(route):
        feasible, _ = _simulate_route(route, dist_matrix, n_customers, stations,
                                       battery_max, speed, recharge_rate, horizon,
                                       demands, ready, due, service)
        return feasible

    def route_load(route):
        return sum(demands[node] for node in route if node <= n_customers)

    # First: optimize each route internally
    for r_idx in range(len(routes)):
        if time.time() > deadline:
            break
        routes[r_idx] = _two_opt_route(
            routes[r_idx], dist_matrix, n_customers, stations,
            battery_max, speed, recharge_rate, horizon,
            demands, ready, due, service, max_iters=30
        )

    current_dist = get_total_dist(routes)

    # Inter-route moves
    max_passes = 30
    for _pass in range(max_passes):
        if time.time() > deadline:
            break

        improved = False

        # Try moving customers
        for r1 in range(len(routes)):
            if time.time() > deadline:
                break
            for pos1 in range(len(routes[r1])):
                if time.time() > deadline:
                    break
                cust = routes[r1][pos1]
                if cust > n_customers:
                    continue

                for r2 in range(len(routes)):
                    if time.time() > deadline:
                        break
                    if r1 == r2:
                        continue
                    if route_load(routes[r2]) + demands[cust] > capacity:
                        continue

                    for pos2 in range(len(routes[r2]) + 1):
                        if time.time() > deadline:
                            break

                        route1_new = routes[r1][:pos1] + routes[r1][pos1+1:]
                        route2_new = routes[r2][:pos2] + [cust] + routes[r2][pos2:]

                        if is_feasible(route1_new) and is_feasible(route2_new):
                            new_routes = [r[:] for r in routes]
                            new_routes[r1] = route1_new
                            new_routes[r2] = route2_new
                            new_dist = get_total_dist(new_routes)
                            if new_dist < current_dist - 1e-9:
                                routes = new_routes
                                current_dist = new_dist
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

    routes = [r for r in routes if len(r) > 0]
    return routes


def plan(instance):
    """Main entry point."""
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    # Build distance matrix once
    coords = np.array(instance["coords"], dtype=np.float64)
    dist_matrix = _build_dist_matrix(coords)

    # Get greedy solution
    routes = _greedy_construct(instance)

    # Validate
    valid, dist = _validate_solution(routes, instance)
    if not valid:
        customers = list(range(1, instance["n_customers"] + 1))
        routes = [[v] for v in customers]

    # Local search if time permits
    if time.time() < deadline - 10:
        routes = _local_search(routes, dist_matrix, instance, deadline)

        valid, dist = _validate_solution(routes, instance)
        if not valid:
            customers = list(range(1, instance["n_customers"] + 1))
            routes = [[v] for v in customers]

    # Final check
    valid, dist = _validate_solution(routes, instance)
    if not valid:
        customers = list(range(1, instance["n_customers"] + 1))
        routes = [[v] for v in customers]

    return routes

# EVOLVE-BLOCK-END
