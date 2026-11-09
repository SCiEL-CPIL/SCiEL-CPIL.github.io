"""Rewritten program for EV routing with time windows and battery constraints."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from itertools import combinations


def _euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _simulate_route(route, instance, coords, demands, ready, due, service,
                   n_customers, stations, capacity, battery_cap, speed,
                   recharge_rate, horizon, depot=0):
    """
    Simulate a single route. Returns (total_distance, feasible).
    route: list of node ids (customers and stations).
    """
    if not route:
        return 0.0, True

    time_cur = 0.0
    battery_cur = battery_cap
    total_dist = 0.0
    total_demand = 0
    prev = depot

    for node in route:
        d = _euclidean(coords[prev], coords[node])
        total_dist += d
        time_cur += d / speed
        battery_cur -= d

        if battery_cur < -1e-9:
            return total_dist, False

        if node in stations:
            # Recharge at station
            if battery_cur < battery_cap:
                time_cur += (battery_cap - battery_cur) * recharge_rate
            battery_cur = battery_cap
        else:
            # Customer node
            total_demand += demands[node]
            if total_demand > capacity + 1e-9:
                return total_dist, False
            # Time window
            if time_cur < ready[node]:
                time_cur = ready[node]
            if time_cur > due[node] + 1e-9:
                return total_dist, False
            time_cur += service[node]

        prev = node

    # Return to depot
    d = _euclidean(coords[prev], coords[depot])
    total_dist += d
    time_cur += d / speed
    battery_cur -= d

    if battery_cur < -1e-9:
        return total_dist, False
    if time_cur > horizon + 1e-9:
        return total_dist, False

    return total_dist, True


def _plan_greedy(instance):
    """
    Greedy construction: nearest feasible neighbor with time-window awareness.
    Returns list of routes.
    """
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    depot = 0

    customers = list(range(1, n_customers + 1))
    remaining = set(customers)
    routes = []

    # Sort customers by due time for initial ordering heuristic
    cust_order = sorted(customers, key=lambda v: due[v])

    while remaining:
        # Start a new route
        route = []
        time_cur = 0.0
        battery_cur = battery_cap
        total_demand = 0
        prev = depot
        total_dist = 0.0

        # Try to add customers greedily
        # We'll use a simple approach: at each step, pick the nearest feasible customer
        # that can be reached and served within constraints.
        while remaining:
            best_node = None
            best_dist = float('inf')
            best_time_after = 0
            best_battery_after = 0

            for c in remaining:
                d = _euclidean(coords[prev], coords[c])
                t_arr = time_cur + d / speed
                b_arr = battery_cur - d
                if b_arr < -1e-9:
                    continue
                if t_arr > due[c] + 1e-9:
                    continue
                if total_demand + demands[c] > capacity + 1e-9:
                    continue
                # Check if we can return to depot after serving c
                # (simplified: just check direct return for now)
                d_back = _euclidean(coords[c], coords[depot])
                b_back = b_arr - d_back
                t_back = t_arr + service[c] + d_back / speed
                if b_back < -1e-9:
                    continue
                if t_back > horizon + 1e-9:
                    continue

                # Feasibility check passed
                if d < best_dist:
                    best_dist = d
                    best_node = c
                    best_time_after = t_arr
                    best_battery_after = b_arr

            if best_node is None:
                break

            # Commit to best_node
            d = _euclidean(coords[prev], coords[best_node])
            time_cur += d / speed
            battery_cur -= d
            total_dist += d

            if time_cur < ready[best_node]:
                time_cur = ready[best_node]
            time_cur += service[best_node]
            total_demand += demands[best_node]

            route.append(best_node)
            remaining.discard(best_node)
            prev = best_node

        if route:
            routes.append(route)
        else:
            # Should not happen if remaining is non-empty, but safety
            break

    return routes


def _improve_routes(routes, instance, max_iter=50, time_limit=None):
    """
    Improve routes using 2-opt and relocate moves.
    """
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    depot = 0

    def total_distance(routes):
        total = 0.0
        for r in routes:
            if not r:
                continue
            prev = depot
            for node in r:
                total += _euclidean(coords[prev], coords[node])
                prev = node
            total += _euclidean(coords[prev], coords[depot])
        return total

    def is_feasible(routes):
        for r in routes:
            _, feas = _simulate_route(r, instance, coords, demands, ready, due,
                                      service, n_customers, stations, capacity,
                                      battery_cap, speed, recharge_rate, horizon, depot)
            if not feas:
                return False
        return True

    best_routes = [list(r) for r in routes]
    best_dist = total_distance(best_routes)
    start_time = time.time()

    for it in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break

        improved = False

        # 2-opt within each route
        for ri, route in enumerate(best_routes):
            n = len(route)
            if n < 2:
                continue
            for i in range(n):
                for j in range(i + 1, n):
                    new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                    test_routes = [list(r) if k != ri else new_route for k, r in enumerate(best_routes)]
                    if is_feasible(test_routes):
                        d = total_distance(test_routes)
                        if d < best_dist - 1e-9:
                            best_routes = test_routes
                            best_dist = d
                            improved = True

        # Relocate: move a customer from one route to another
        for ri in range(len(best_routes)):
            for ci in range(len(best_routes[ri])):
                cust = best_routes[ri][ci]
                for rj in range(len(best_routes)):
                    if rj == ri:
                        continue
                    for cj in range(len(best_routes[rj]) + 1):
                        # Try inserting cust after position cj-1 in route rj
                        new_ri = list(best_routes[ri])
                        new_ri.pop(ci)
                        new_rj = list(best_routes[rj])
                        new_rj.insert(cj, cust)
                        test_routes = []
                        for k in range(len(best_routes)):
                            if k == ri:
                                test_routes.append(new_ri)
                            elif k == rj:
                                test_routes.append(new_rj)
                            else:
                                test_routes.append(list(best_routes[k]))
                        if is_feasible(test_routes):
                            d = total_distance(test_routes)
                            if d < best_dist - 1e-9:
                                best_routes = test_routes
                                best_dist = d
                                improved = True
                                ci = -1  # reset inner loop
                                break
                    if ci == -1:
                        break

        if not improved:
            break

    return best_routes


def plan(instance):
    """
    Main entry point. Returns list of routes for the given instance.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    depot = 0

    # Phase 1: Greedy construction
    routes = _plan_greedy(instance)

    # Verify all customers are served
    all_served = set()
    for r in routes:
        for node in r:
            if node not in set(stations):
                all_served.add(node)
    if all_served != set(range(1, n_customers + 1)):
        # Fallback: one customer per route
        routes = [[v] for v in range(1, n_customers + 1)]

    # Phase 2: Improvement
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        max_iter = min(50, int(remaining_time / 0.5))
        routes = _improve_routes(routes, instance, max_iter=max_iter,
                                 time_limit=remaining_time - 5.0)

    # Final feasibility check
    final_routes = []
    for r in routes:
        _, feas = _simulate_route(r, instance, coords, demands, ready, due,
                                  service, n_customers, stations, capacity,
                                  battery_cap, speed, recharge_rate, horizon, depot)
        if feas:
            final_routes.append(r)
        else:
            # Split into individual routes
            for node in r:
                final_routes.append([node])

    # Ensure all customers are covered
    all_served = set()
    for r in final_routes:
        for node in r:
            if node not in set(stations):
                all_served.add(node)
    missing = set(range(1, n_customers + 1)) - all_served
    for m in missing:
        final_routes.append([m])

    return final_routes
# EVOLVE-BLOCK-END
