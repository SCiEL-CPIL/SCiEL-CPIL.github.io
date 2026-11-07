"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def _euclid(coords, i, j):
    dx = coords[i][0] - coords[j][0]
    dy = coords[i][1] - coords[j][1]
    return math.sqrt(dx * dx + dy * dy)


def _build_dist_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _euclid(coords, i, j)
            D[i][j] = d
            D[j][i] = d
    return D


def _simulate_route(route, D, instance):
    """Simulate a route (list of node ids, no depot). Return (total_dist, valid, end_time, end_battery)."""
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

    # Check capacity
    total_demand = sum(demands[v] for v in route if v not in stations)
    if total_demand > capacity:
        return 0, False, 0, 0

    t = 0.0
    battery = battery_max
    total_dist = 0.0
    prev = 0  # depot

    for v in route:
        d = D[prev][v]
        t += d / speed
        battery -= d
        total_dist += d
        if battery < -1e-9:
            return 0, False, 0, 0

        if v in stations:
            # Recharge: time added is (battery_max - battery) * recharge_rate... wait
            # "a station adds (battery - energy) * recharge_rate time and refills"
            # I think this means: the time to recharge is (battery_max - battery) / recharge_rate? 
            # No, re-reading: "a station adds (battery - energy) * recharge_rate time and refills"
            # Hmm, "battery - energy" is ambiguous. Let me re-read the task.
            # "a station adds (battery - energy) * recharge_rate time and refills; stations may repeat"
            # I think "energy" here might mean the current battery level? Or maybe it's a typo.
            # Actually, looking at typical EV routing: recharge time = (capacity - current) / rate.
            # But the task says "(battery - energy) * recharge_rate". 
            # Given recharge_rate=0.5, if it's (300 - battery) * 0.5, that's a lot of time.
            # Let me assume: recharge_time = (battery_max - battery) / recharge_rate? No, that would be division.
            # The text says "* recharge_rate". So maybe recharge_time = (battery_max - battery) * recharge_rate?
            # With recharge_rate=0.5, if battery=0, time = 300*0.5 = 150. That seems high but possible.
            # Actually, let me reconsider. Maybe "energy" is a fixed value? No.
            # Let me just implement it as: recharge_time = (battery_max - battery) * recharge_rate
            # and then battery = battery_max.
            recharge_time = (battery_max - battery) * recharge_rate
            t += recharge_time
            battery = battery_max
        else:
            # Customer
            if t < ready[v]:
                t = ready[v]
            if t > due[v] + 1e-9:
                return 0, False, 0, 0
            t += service[v]

        prev = v

    # Return to depot
    d = D[prev][0]
    t += d / speed
    battery -= d
    total_dist += d
    if battery < -1e-9:
        return 0, False, 0, 0
    if t > horizon + 1e-9:
        return 0, False, 0, 0

    return total_dist, True, t, battery


def _total_distance(routes, D):
    total = 0.0
    for route in routes:
        if not route:
            continue
        prev = 0
        for v in route:
            total += D[prev][v]
            prev = v
        total += D[prev][0]
    return total


def _validate_all(routes, instance, D):
    """Check all constraints. Return True if valid."""
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

    # Check every customer served exactly once
    served = set()
    for route in routes:
        for v in route:
            if v in stations:
                continue
            if v in served:
                return False
            served.add(v)
    if len(served) != n_customers:
        return False

    # Check each route
    for route in routes:
        total_demand = sum(demands[v] for v in route if v not in stations)
        if total_demand > capacity:
            return False

        t = 0.0
        battery = battery_max
        prev = 0

        for v in route:
            d = D[prev][v]
            t += d / speed
            battery -= d
            if battery < -1e-9:
                return False

            if v in stations:
                recharge_time = (battery_max - battery) * recharge_rate
                t += recharge_time
                battery = battery_max
            else:
                if t < ready[v]:
                    t = ready[v]
                if t > due[v] + 1e-9:
                    return False
                t += service[v]

            prev = v

        d = D[prev][0]
        t += d / speed
        battery -= d
        if battery < -1e-9:
            return False
        if t > horizon + 1e-9:
            return False

    return True


def _savings_heuristic(instance, D, seed=0):
    """Clarke-Wright savings heuristic with time/battery feasibility checks."""
    rng = random.Random(seed)
    customers = list(range(1, instance["n_customers"] + 1))
    stations = set(instance["stations"])
    n = instance["n_customers"]

    # Initial: each customer as separate route
    routes = {c: [c] for c in customers}

    # Compute savings: D[0][i] + D[0][j] - D[i][j]
    savings = []
    for i in customers:
        for j in customers:
            if i < j:
                s = D[0][i] + D[0][j] - D[i][j]
                savings.append((s, i, j))
    savings.sort(reverse=True)

    route_of = {c: c for c in customers}

    for s, i, j in savings:
        ri = route_of[i]
        rj = route_of[j]
        if ri == rj:
            continue
        if s <= 0:
            break

        # Try to merge: i and j must be at the ends of their routes
        ri_list = routes[ri]
        rj_list = routes[rj]

        # Check if i is at an end of ri_list
        i_end = None
        if ri_list[0] == i:
            i_end = 0
        elif ri_list[-1] == i:
            i_end = 1

        j_end = None
        if rj_list[0] == j:
            j_end = 0
        elif rj_list[-1] == j:
            j_end = 1

        if i_end is None or j_end is None:
            continue

        # Build merged route
        if i_end == 0:
            ri_part = ri_list[::-1]  # i becomes last
        else:
            ri_part = ri_list  # i is last

        if j_end == 0:
            rj_part = rj_list  # j is first
        else:
            rj_part = rj_list[::-1]  # j becomes first

        merged = ri_part + rj_part

        # Check feasibility
        dist, valid, _, _ = _simulate_route(merged, D, instance)
        if valid:
            routes[ri] = merged
            del routes[rj]
            for c in rj_list:
                route_of[c] = ri

    return list(routes.values())


def _nearest_neighbor_heuristic(instance, D, seed=0):
    """Nearest neighbor heuristic with feasibility checks."""
    rng = random.Random(seed)
    customers = list(range(1, instance["n_customers"] + 1))
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]

    routes = []
    unvisited = set(customers)

    while unvisited:
        # Start from a random unvisited customer
        start = rng.choice(list(unvisited))
        route = [start]
        unvisited.remove(start)
        total_demand = demands[start]
        t = 0.0
        battery = battery_max
        prev = 0

        # Travel to start
        d = D[0][start]
        t += d / speed
        battery -= d
        if battery < 0:
            continue
        if t < ready[start]:
            t = ready[start]
        if t > due[start]:
            continue
        t += service[start]

        while True:
            best_next = None
            best_dist = float('inf')
            for c in unvisited:
                if total_demand + demands[c] > capacity:
                    continue
                d = D[prev][c]
                if d >= best_dist:
                    continue
                # Simulate adding c
                t_new = t + d / speed
                battery_new = battery - d
                if battery_new < -1e-9:
                    continue
                if t_new < ready[c]:
                    t_new = ready[c]
                if t_new > due[c]:
                    continue
                t_new += service[c]
                # Check return to depot
                d_return = D[c][0]
                t_final = t_new + d_return / speed
                battery_final = battery_new - d_return
                if battery_final < -1e-9:
                    continue
                if t_final > horizon:
                    continue
                best_dist = d
                best_next = c

            if best_next is None:
                break

            route.append(best_next)
            unvisited.remove(best_next)
            total_demand += demands[best_next]
            d = D[prev][best_next]
            t += d / speed
            battery -= d
            if t < ready[best_next]:
                t = ready[best_next]
            t += service[best_next]
            prev = best_next

        if route:
            # Final check
            dist, valid, _, _ = _simulate_route(route, D, instance)
            if valid:
                routes.append(route)
            else:
                # Re-add customers
                for c in route:
                    unvisited.add(c)

    return routes


def _local_search_2opt_or_opt(instance, D, routes, max_iters=500, seed=0):
    """2-opt and Or-opt local search."""
    rng = random.Random(seed)
    stations = set(instance["stations"])
    improved = True
    iters = 0

    while improved and iters < max_iters:
        improved = False
        iters += 1

        for ri, route in enumerate(routes):
            if len(route) < 2:
                continue

            # 2-opt
            for i in range(len(route) - 1):
                for j in range(i + 1, len(route)):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    dist_old, valid_old, _, _ = _simulate_route(route, D, instance)
                    dist_new, valid_new, _, _ = _simulate_route(new_route, D, instance)
                    if valid_new and dist_new < dist_old - 1e-9:
                        routes[ri] = new_route
                        improved = True

            # Or-opt: move 1 or 2 consecutive nodes
            for i in range(len(route)):
                for length in [1, 2]:
                    if i + length > len(route):
                        continue
                    segment = route[i:i+length]
                    rest = route[:i] + route[i+length:]
                    for j in range(len(rest) + 1):
                        new_route = rest[:j] + segment + rest[j:]
                        dist_old, valid_old, _, _ = _simulate_route(route, D, instance)
                        dist_new, valid_new, _, _ = _simulate_route(new_route, D, instance)
                        if valid_new and dist_new < dist_old - 1e-9:
                            routes[ri] = new_route
                            improved = True
                            break

    return routes


def _inter_route_relocate(instance, D, routes, max_iters=100, seed=0):
    """Try to relocate customers between routes."""
    rng = random.Random(seed)
    stations = set(instance["stations"])
    improved = True
    iters = 0

    while improved and iters < max_iters:
        improved = False
        iters += 1

        for ri in range(len(routes)):
            for i, c in enumerate(routes[ri]):
                if c in stations:
                    continue
                for rj in range(len(routes)):
                    if rj == ri:
                        continue
                    for j in range(len(routes[rj]) + 1):
                        # Remove c from route ri
                        new_ri = routes[ri][:i] + routes[ri][i+1:]
                        # Insert c into route rj at position j
                        new_rj = routes[rj][:j] + [c] + routes[rj][j:]

                        dist_old_ri, valid_old_ri, _, _ = _simulate_route(routes[ri], D, instance)
                        dist_old_rj, valid_old_rj, _, _ = _simulate_route(routes[rj], D, instance)
                        dist_new_ri, valid_new_ri, _, _ = _simulate_route(new_ri, D, instance)
                        dist_new_rj, valid_new_rj, _, _ = _simulate_route(new_rj, D, instance)

                        if valid_new_ri and valid_new_rj:
                            total_old = dist_old_ri + dist_old_rj
                            total_new = dist_new_ri + dist_new_rj
                            if total_new < total_old - 1e-9:
                                routes[ri] = new_ri
                                routes[rj] = new_rj
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
    deadline = start_time + 140.0  # 10s safety margin

    coords = instance["coords"]
    D = _build_dist_matrix(coords)
    n_customers = instance["n_customers"]

    best_routes = None
    best_dist = float('inf')

    # Generate multiple initial solutions using different heuristics and seeds
    heuristics = []

    # Savings heuristic with different seeds
    for seed in range(5):
        routes = _savings_heuristic(instance, D, seed=seed)
        if _validate_all(routes, instance, D):
            dist = _total_distance(routes, D)
            heuristics.append((dist, routes, f"savings_{seed}"))
            if dist < best_dist:
                best_dist = dist
                best_routes = [list(r) for r in routes]

    # Nearest neighbor with different seeds
    for seed in range(5):
        routes = _nearest_neighbor_heuristic(instance, D, seed=seed)
        if _validate_all(routes, instance, D):
            dist = _total_distance(routes, D)
            heuristics.append((dist, routes, f"nn_{seed}"))
            if dist < best_dist:
                best_dist = dist
                best_routes = [list(r) for r in routes]

    # Sort by distance
    heuristics.sort(key=lambda x: x[0])

    # Local search on the best few solutions
    for dist, routes, name in heuristics[:3]:
        if time.time() > deadline - 20:
            break
        routes_copy = [list(r) for r in routes]
        routes_copy = _local_search_2opt_or_opt(instance, D, routes_copy, max_iters=200, seed=42)
        if _validate_all(routes_copy, instance, D):
            new_dist = _total_distance(routes_copy, D)
            if new_dist < best_dist:
                best_dist = new_dist
                best_routes = [list(r) for r in routes_copy]

        # Inter-route relocate
        if time.time() > deadline - 10:
            break
        routes_copy = [list(r) for r in routes_copy]
        routes_copy = _inter_route_relocate(instance, D, routes_copy, max_iters=50, seed=42)
        if _validate_all(routes_copy, instance, D):
            new_dist = _total_distance(routes_copy, D)
            if new_dist < best_dist:
                best_dist = new_dist
                best_routes = [list(r) for r in routes_copy]

    # Final local search on the best solution
    if best_routes is not None and time.time() < deadline - 5:
        best_routes = _local_search_2opt_or_opt(instance, D, best_routes, max_iters=100, seed=123)
        if _validate_all(best_routes, instance, D):
            new_dist = _total_distance(best_routes, D)
            if new_dist < best_dist:
                best_dist = new_dist

    if best_routes is None:
        # Fallback: one customer per route
        customers = list(range(1, n_customers + 1))
        best_routes = [[c] for c in customers]

    return best_routes
# EVOLVE-BLOCK-END
