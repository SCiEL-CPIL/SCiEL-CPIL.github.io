"""EV routing with time windows: constructive nearest-neighbour + 2-opt/relocate/or-opt local search."""
# EVOLVE-BLOCK-START
import math
import random
import numpy as np


def _route_distance(route, dist):
    if not route:
        return 0.0
    total = dist[0][route[0]]
    for i in range(len(route) - 1):
        total += dist[route[i]][route[i + 1]]
    total += dist[route[-1]][0]
    return total


def _check_route(route, instance, dist):
    """Validate one route (depot implicit at both ends). Returns (valid, total_distance)."""
    cap = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations_set = instance["_stations_set"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]

    total_demand = 0
    total_dist = 0.0
    battery = battery_max
    time = 0.0
    prev = 0
    for node in route:
        d = dist[prev][node]
        total_dist += d
        time += d / speed
        battery -= d
        if battery < -1e-9:
            return False, total_dist
        if node not in stations_set:
            # customer
            if time > due[node] + 1e-9:
                return False, total_dist
            if time < ready[node]:
                time = ready[node]
            time += service[node]
            total_demand += demands[node]
            if total_demand > cap:
                return False, total_dist
        else:
            # station
            energy = battery_max - battery
            if energy > 0:
                time += energy * recharge_rate
            battery = battery_max
        prev = node
    d = dist[prev][0]
    total_dist += d
    time += d / speed
    battery -= d
    if battery < -1e-9:
        return False, total_dist
    if time > horizon + 1e-9:
        return False, total_dist
    return True, total_dist


def _solve_instance(instance, seed):
    rng = random.Random(seed)
    coords = instance["coords"]
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    stations = set(instance["stations"])
    instance["_stations_set"] = stations
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    n_nodes = len(coords)
    dist = np.zeros((n_nodes, n_nodes), dtype=float)
    for i in range(n_nodes):
        xi, yi = coords[i]
        for j in range(i + 1, n_nodes):
            xj, yj = coords[j]
            dd = math.hypot(xi - xj, yi - yj)
            dist[i, j] = dd
            dist[j, i] = dd

    order = sorted(customers, key=lambda v: (due[v], ready[v]))

    routes = []

    def try_insert(route, pos, node):
        new_route = route[:pos] + [node] + route[pos:]
        total = sum(demands[n] for n in new_route if n not in stations)
        if total > capacity:
            return False, None
        valid, _ = _check_route(new_route, instance, dist)
        return valid, new_route

    for node in order:
        best_cost = None
        best_choice = None
        for ri, route in enumerate(routes):
            old_cost = _route_distance(route, dist)
            for pos in range(len(route) + 1):
                valid, new_route = try_insert(route, pos, node)
                if not valid:
                    continue
                new_cost = _route_distance(new_route, dist)
                delta = new_cost - old_cost
                if best_cost is None or delta < best_cost - 1e-9:
                    best_cost = delta
                    best_choice = (ri, new_route)
        if best_choice is not None:
            ri, new_route = best_choice
            routes[ri] = new_route
        else:
            valid, _ = _check_route([node], instance, dist)
            if valid:
                routes.append([node])
            else:
                placed = False
                for st in stations:
                    for candidate in ([st, node], [node, st]):
                        valid, _ = _check_route(candidate, instance, dist)
                        if valid:
                            routes.append(candidate)
                            placed = True
                            break
                    if placed:
                        break
                if not placed:
                    routes.append([node])

    # Local search
    improved = True
    iters = 0
    max_iters = 50
    while improved and iters < max_iters:
        improved = False
        iters += 1

        # 2-opt within each route
        for ri in range(len(routes)):
            route = routes[ri]
            if len(route) < 2:
                continue
            best_delta = 1e-9
            best_route = None
            for i in range(len(route)):
                for j in range(i + 1, len(route)):
                    new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                    valid, _ = _check_route(new_route, instance, dist)
                    if not valid:
                        continue
                    old_c = _route_distance(route, dist)
                    new_c = _route_distance(new_route, dist)
                    delta = new_c - old_c
                    if delta < best_delta - 1e-9:
                        best_delta = delta
                        best_route = new_route
            if best_route is not None:
                routes[ri] = best_route
                improved = True

        # Relocate single customer
        for ri in range(len(routes)):
            route = routes[ri]
            for ni in range(len(route)):
                node = route[ni]
                if node in stations:
                    continue
                route_wo = route[:ni] + route[ni + 1:]
                for rj in range(len(routes)):
                    if rj == ri:
                        continue
                    route2 = routes[rj]
                    for pos in range(len(route2) + 1):
                        new_route2 = route2[:pos] + [node] + route2[pos:]
                        valid1, _ = _check_route(route_wo, instance, dist)
                        valid2, _ = _check_route(new_route2, instance, dist)
                        if not valid1 or not valid2:
                            continue
                        old_c = _route_distance(route, dist) + _route_distance(route2, dist)
                        new_c = _route_distance(route_wo, dist) + _route_distance(new_route2, dist)
                        if new_c < old_c - 1e-9:
                            routes[ri] = route_wo
                            routes[rj] = new_route2
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

        # Or-opt: move 1-2 consecutive customers
        for ri in range(len(routes)):
            route = routes[ri]
            for ni in range(len(route)):
                for seg_len in [1, 2]:
                    if ni + seg_len > len(route):
                        continue
                    seg = route[ni:ni + seg_len]
                    if any(n in stations for n in seg):
                        continue
                    route_wo = route[:ni] + route[ni + seg_len:]
                    for rj in range(len(routes)):
                        if rj == ri:
                            continue
                        route2 = routes[rj]
                        for pos in range(len(route2) + 1):
                            new_route2 = route2[:pos] + seg + route2[pos:]
                            valid1, _ = _check_route(route_wo, instance, dist)
                            valid2, _ = _check_route(new_route2, instance, dist)
                            if not valid1 or not valid2:
                                continue
                            old_c = _route_distance(route, dist) + _route_distance(route2, dist)
                            new_c = _route_distance(route_wo, dist) + _route_distance(new_route2, dist)
                            if new_c < old_c - 1e-9:
                                routes[ri] = route_wo
                                routes[rj] = new_route2
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

        routes = [r for r in routes if r]

    # Final validation
    result = []
    all_customers = set(customers)
    for route in routes:
        valid, _ = _check_route(route, instance, dist)
        if valid:
            result.append(route)
        else:
            for node in route:
                if node in all_customers:
                    result.append([node])

    covered = set()
    for route in result:
        for node in route:
            covered.add(node)
    missing = all_customers - covered
    for node in sorted(missing, key=lambda v: due[v]):
        result.append([node])

    return result


def plan(instance):
    seed = 42
    routes = _solve_instance(instance, seed)
    return routes
# EVOLVE-BLOCK-END
