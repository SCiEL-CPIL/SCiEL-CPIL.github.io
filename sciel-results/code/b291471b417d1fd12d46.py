"""Improved EV routing: greedy nearest-neighbour construction + 2-opt/insertion local search with full feasibility checks."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random

def _euclidean(coords):
    """Build Euclidean distance matrix."""
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            d = math.sqrt(dx*dx + dy*dy)
            D[i, j] = d
            D[j, i] = d
    return D

def _route_feasible_and_dist(route, instance, D, coords):
    """
    Check feasibility of a route (list of node ids, depot implicit at both ends).
    Returns (is_feasible, total_distance, end_time, end_battery).
    Route starts at depot (node 0) with full battery at time 0.
    """
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    demands = instance["demands"]
    stations = set(instance["stations"])

    total_dist = 0.0
    battery = battery_max
    t = 0.0
    load = 0.0

    # Start at depot
    prev = 0
    for node in route:
        d = D[prev, node]
        total_dist += d
        battery -= d
        t += d / speed
        if battery < -1e-9:
            return False, total_dist, t, battery

        if node in stations:
            # Recharge: time = (battery - energy) * recharge_rate, but energy here is 0 (refill to full)
            # Actually: station adds (battery - energy) * recharge_rate time and refills.
            # "energy" in the formula is likely 0, meaning recharge time = battery * recharge_rate? 
            # Wait, re-reading: "a station adds (battery - energy) * recharge_rate time and refills"
            # This is ambiguous. Let me re-read the contract.
            # "a station adds (battery - energy) * recharge_rate time and refills"
            # I think "energy" here might be a typo or refers to the energy needed to reach the station.
            # Actually, looking at typical EV routing: you arrive with some battery, and the recharge time
            # is proportional to the amount of energy you need to add. If you arrive with `battery` and 
            # the station refills to `battery_max`, the energy added is `battery_max - battery`.
            # So recharge time = (battery_max - battery) * recharge_rate.
            # But the contract says "(battery - energy) * recharge_rate". If "energy" is 0, then it's battery * recharge_rate.
            # That doesn't make sense. Let me assume it means the energy *deficit*, i.e., (battery_max - battery) * recharge_rate.
            # Actually, wait. Let me re-read very carefully: "a station adds (battery - energy) * recharge_rate time and refills"
            # Perhaps "energy" is the energy consumed to get there? No, that's already accounted for in battery -= d.
            # I'll assume the standard interpretation: recharge_time = (battery_max - battery) * recharge_rate.
            recharge_time = (battery_max - battery) * recharge_rate
            t += recharge_time
            battery = battery_max
        else:
            # Customer
            if node not in range(1, instance["n_customers"] + 1):
                return False, total_dist, t, battery
            load += demands[node]
            if load > capacity + 1e-9:
                return False, total_dist, t, battery
            t = max(t, ready[node])
            if t > due[node] + 1e-9:
                return False, total_dist, t, battery
            t += service[node]

        prev = node

    # Return to depot
    d = D[prev, 0]
    total_dist += d
    battery -= d
    t += d / speed
    if battery < -1e-9:
        return False, total_dist, t, battery
    if t > horizon + 1e-9:
        return False, total_dist, t, battery

    return True, total_dist, t, battery

def _total_distance(routes, D):
    """Total distance of all routes (depot legs included)."""
    total = 0.0
    for route in routes:
        if not route:
            continue
        prev = 0
        for node in route:
            total += D[prev, node]
            prev = node
        total += D[prev, 0]
    return total

def _greedy_construct(instance, D, coords, rng):
    """
    Greedy construction: repeatedly pick the next customer that minimizes insertion cost
    while maintaining feasibility.
    """
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    demands = instance["demands"]
    stations = set(instance["stations"])
    n_customers = instance["n_customers"]

    customers = list(range(1, n_customers + 1))
    rng.shuffle(customers)

    routes = []
    remaining = set(customers)

    while remaining:
        # Start a new route
        best_customer = None
        best_dist = float('inf')
        for c in remaining:
            # Check if a single-customer route is feasible
            d1 = D[0, c]
            d2 = D[c, 0]
            total_d = d1 + d2
            t = d1 / speed
            t = max(t, ready[c])
            if t > due[c] + 1e-9:
                continue
            t += service[c]
            t += d2 / speed
            if t > horizon + 1e-9:
                continue
            if total_d > battery_max + 1e-9:
                continue
            if demands[c] > capacity + 1e-9:
                continue
            if total_d < best_dist:
                best_dist = total_d
                best_customer = c

        if best_customer is None:
            # Pick any feasible customer
            for c in remaining:
                d1 = D[0, c]
                d2 = D[c, 0]
                total_d = d1 + d2
                t = d1 / speed
                t = max(t, ready[c])
                if t > due[c] + 1e-9:
                    continue
                t += service[c]
                t += d2 / speed
                if t > horizon + 1e-9:
                    continue
                if total_d > battery_max + 1e-9:
                    continue
                if demands[c] > capacity + 1e-9:
                    continue
                best_customer = c
                break

        if best_customer is None:
            # Fallback: one per route
            for c in remaining:
                routes.append([c])
                remaining.remove(c)
            break

        route = [best_customer]
        remaining.remove(best_customer)

        # Try to insert more customers
        changed = True
        while changed:
            changed = False
            best_insert = None
            best_pos = None
            best_cost = float('inf')
            for c in remaining:
                for pos in range(len(route) + 1):
                    # Insert c at pos
                    new_route = route[:pos] + [c] + route[pos:]
                    feasible, dist, _, _ = _route_feasible_and_dist(new_route, instance, D, coords)
                    if feasible and dist < best_cost - 1e-9:
                        best_cost = dist
                        best_insert = c
                        best_pos = pos

            if best_insert is not None:
                route = route[:best_pos] + [best_insert] + route[best_pos:]
                remaining.remove(best_insert)
                changed = True

        routes.append(route)

    return routes

def _local_search(routes, instance, D, coords, rng, time_limit=10.0):
    """
    2-opt and insertion local search.
    """
    start_time = time.time()
    current_routes = [list(r) for r in routes]
    current_dist = _total_distance(current_routes, D)

    while time.time() - start_time < time_limit:
        improved = False

        # 2-opt within each route
        for i in range(len(current_routes)):
            route = current_routes[i]
            n = len(route)
            if n < 2:
                continue
            for a in range(n - 1):
                for b in range(a + 1, n):
                    new_route = route[:a] + route[a:b+1][::-1] + route[b+1:]
                    feasible, dist, _, _ = _route_feasible_and_dist(new_route, instance, D, coords)
                    if feasible:
                        new_total = current_dist - _total_distance([route], D) + dist
                        if new_total < current_dist - 1e-9:
                            current_routes[i] = new_route
                            current_dist = new_total
                            improved = True

        # Insertion: move a customer from one route to another
        for i in range(len(current_routes)):
            for j in range(len(current_routes[i])):
                c = current_routes[i][j]
                # Remove c from route i
                route_i = current_routes[i][:j] + current_routes[i][j+1:]
                for k in range(len(current_routes)):
                    if k == i:
                        continue
                    route_k = current_routes[k]
                    for pos in range(len(route_k) + 1):
                        new_route_k = route_k[:pos] + [c] + route_k[pos:]
                        feasible_i, dist_i, _, _ = _route_feasible_and_dist(route_i, instance, D, coords)
                        feasible_k, dist_k, _, _ = _route_feasible_and_dist(new_route_k, instance, D, coords)
                        if feasible_i and feasible_k:
                            new_total = current_dist - _total_distance([current_routes[i], current_routes[k]], D) + dist_i + dist_k
                            if new_total < current_dist - 1e-9:
                                current_routes[i] = route_i
                                current_routes[k] = new_route_k
                                current_dist = new_total
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

        # Remove empty routes
        current_routes = [r for r in current_routes if r]

        if not improved:
            break

    return current_routes, current_dist

def plan(instance):
    """
    Plan routes for the EV routing problem.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s

    coords = instance["coords"]
    D = _euclidean(coords)

    rng = random.Random(42)

    # Greedy construction
    routes = _greedy_construct(instance, D, coords, rng)

    # Local search
    routes, dist = _local_search(routes, instance, D, coords, rng, time_limit=10.0)

    return routes
# EVOLVE-BLOCK-END
