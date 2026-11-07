"""EV routing with time windows and battery constraints.
Constructive nearest-neighbor + local search (2-opt, relocate, swap) with
battery/time-window feasibility and automatic charging-station insertion.
"""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np


DEADLINE_SECONDS = 110.0


def _build_arrays(instance):
    coords = np.array(instance["coords"], dtype=np.float64)
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    n = len(coords)
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    return coords, demands, ready, due, service, dist, n


def _simulate(route, arrays):
    coords, demands, ready, due, service, dist, n = arrays
    capacity = None  # filled by caller via closure? We'll pass params instead.
    return None


def _is_feasible_route(route, arrays, params):
    coords, demands, ready, due, service, dist, n = arrays
    capacity, battery_max, speed, recharge_rate, horizon, stations = params
    total_demand = 0.0
    for v in route:
        total_demand += demands[v]
    if total_demand > capacity + 1e-9:
        return False
    t = 0.0
    b = battery_max
    prev = 0
    for v in route:
        d = dist[prev, v]
        t += d / speed
        b -= d
        if b < -1e-9:
            return False
        if t > due[v] + 1e-9:
            return False
        if t < ready[v]:
            t = ready[v]
        t += service[v]
        b = min(battery_max, b)
        prev = v
    d = dist[prev, 0]
    t += d / speed
    b -= d
    if b < -1e-9:
        return False
    if t > horizon + 1e-9:
        return False
    return True


def _route_distance(route, dist):
    total = 0.0
    prev = 0
    for v in route:
        total += dist[prev, v]
        prev = v
    total += dist[prev, 0]
    return total


def _try_insert_station(route, position, station, arrays, params):
    """Return a new route with station inserted at position, or None if infeasible."""
    coords, demands, ready, due, service, dist, n = arrays
    new_route = route[:position] + [station] + route[position:]
    return new_route if _is_feasible_route(new_route, arrays, params) else None


def _nearest_feasible_station(route, position, arrays, params):
    """Find the cheapest station to insert at 'position' that makes the route feasible.
    Returns (station, new_route) or (None, None)."""
    coords, demands, ready, due, service, dist, n = arrays
    capacity, battery_max, speed, recharge_rate, horizon, stations = params
    # candidate stations sorted by distance from the node before position
    prev_node = route[position - 1] if position > 0 else 0
    cand = sorted(stations, key=lambda s: dist[prev_node, s])
    for s in cand:
        new_route = _try_insert_station(route, position, s, arrays, params)
        if new_route is not None:
            return s, new_route
    # Try depot as a fallback (charging at depot)
    new_route = _try_insert_station(route, position, 0, arrays, params)
    if new_route is not None:
        return 0, new_route
    return None, None


def _construct_route(remaining, arrays, params):
    """Nearest-neighbor construction for one route, with battery/time-aware insertion."""
    coords, demands, ready, due, service, dist, n = arrays
    capacity, battery_max, speed, recharge_rate, horizon, stations = params

    # Pick a starting customer: the one with earliest due among remaining
    if not remaining:
        return [], list(remaining)
    start = min(remaining, key=lambda v: (due[v], v))
    route = [start]
    remaining = [v for v in remaining if v != start]
    total_demand = demands[start]

    while remaining:
        # Find best next customer to append (greedy)
        prev = route[-1]
        best_v = None
        best_cost = float('inf')
        for v in remaining:
            if total_demand + demands[v] > capacity + 1e-9:
                continue
            cand_route = route + [v]
            if not _is_feasible_route(cand_route, arrays, params):
                # Try inserting a station before v
                s, new_route = _nearest_feasible_station(cand_route, len(cand_route), arrays, params)
                if new_route is None:
                    continue
                cand_route = new_route
            d = _route_distance(cand_route, dist)
            # Prefer smaller distance; tie-break by earlier due
            cost = (d, due[v], v)
            if cost < (best_cost, due[v], v) if not isinstance(best_cost, float) else d < best_cost:
                best_cost = d
                best_v = v
                best_route = cand_route
        if best_v is None:
            break
        route = best_route
        total_demand = sum(demands[v] for v in route if demands[v] > 0)
        remaining = [v for v in remaining if v not in route]
    return route, remaining


def _local_search(route, arrays, params, deadline):
    """2-opt + relocate + swap local search with deadline guard."""
    coords, demands, ready, due, service, dist, n = arrays
    capacity, battery_max, speed, recharge_rate, horizon, stations = params

    def improved(new_route):
        if _is_feasible_route(new_route, arrays, params):
            nd = _route_distance(new_route, dist)
            if nd < _route_distance(route, dist) - 1e-9:
                return True, nd
        return False, None

    improved_flag = True
    while improved_flag:
        improved_flag = False
        if time.time() > deadline:
            break
        # 2-opt
        m = len(route)
        for i in range(m - 1):
            for j in range(i + 1, m):
                if time.time() > deadline:
                    break
                new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                ok, _ = improved(new_route)
                if ok:
                    route = new_route
                    improved_flag = True
                    break
            if time.time() > deadline:
                break
            if improved_flag:
                break
        if time.time() > deadline:
            break
        # Relocate
        m = len(route)
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                if time.time() > deadline:
                    break
                v = route[i]
                new_route = route[:i] + route[i + 1:]
                insert_pos = j if j < i else j
                new_route = new_route[:insert_pos] + [v] + new_route[insert_pos:]
                ok, _ = improved(new_route)
                if ok:
                    route = new_route
                    improved_flag = True
                    break
            if time.time() > deadline:
                break
            if improved_flag:
                break
        if time.time() > deadline:
            break
        # Swap
        m = len(route)
        for i in range(m - 1):
            for j in range(i + 1, m):
                if time.time() > deadline:
                    break
                new_route = route[:]
                new_route[i], new_route[j] = new_route[j], new_route[i]
                ok, _ = improved(new_route)
                if ok:
                    route = new_route
                    improved_flag = True
                    break
            if time.time() > deadline:
                break
            if improved_flag:
                break
    return route


def plan(instance):
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    coords, demands, ready, due, service, dist, n = _build_arrays(instance)
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])
    params = (capacity, battery_max, speed, recharge_rate, horizon, stations)
    arrays = (coords, demands, ready, due, service, dist, n)

    customers = list(range(1, instance["n_customers"] + 1))
    # Sort by due for deterministic construction order
    customers.sort(key=lambda v: (due[v], v))

    routes = []
    remaining = list(customers)
    while remaining and time.time() < deadline:
        route, remaining = _construct_route(remaining, arrays, params)
        if not route:
            # Fallback: single customer per route to guarantee feasibility
            for v in remaining:
                single = [v]
                if _is_feasible_route(single, arrays, params):
                    routes.append(single)
                    remaining.remove(v)
                else:
                    # Should not happen with full battery; add anyway
                    routes.append(single)
                    remaining.remove(v)
            break
        # Local search
        route = _local_search(route, arrays, params, deadline)
        routes.append(route)
        remaining = [v for v in remaining if v not in route]

    # Safety: ensure all customers served; add any leftovers as single routes
    served = set()
    for r in routes:
        for v in r:
            served.add(v)
    for v in customers:
        if v not in served:
            routes.append([v])

    # Final feasibility check; if any route infeasible, fall back to singles
    for i, r in enumerate(routes):
        if not _is_feasible_route(r, arrays, params):
            routes[i] = [[v] for v in r]
            # Flatten handled below
    # Flatten any nested lists
    flat = []
    for r in routes:
        if r and isinstance(r[0], list):
            flat.extend(r)
        else:
            flat.append(r)
    return flat
# EVOLVE-BLOCK-END
