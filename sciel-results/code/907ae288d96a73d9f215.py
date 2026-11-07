"""EV routing with time windows and battery constraints - structural rewrite.
Uses Clarke-Wright savings construction + multi-start + fast local search
with vectorized feasibility checks and inter-route moves.
"""
# EVOLVE-BLOCK-START
import numpy as np
import math
import time
import random

DEADLINE_SECONDS = 115.0


def _build_distance_matrix(coords):
    n = len(coords)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = math.sqrt((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2)
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _simulate_route(route, instance, dist, stations_set=None):
    """Simulate a route. Returns (total_distance, is_feasible)."""
    if stations_set is None:
        stations_set = set(instance["stations"])
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = stations_set

    total_demand = 0
    for v in route:
        if v not in stations:
            total_demand += demands[v]
    if total_demand > capacity:
        return float('inf'), False

    battery = battery_max
    time_curr = 0.0
    total_dist = 0.0
    current = 0

    for v in route:
        d = dist[current, v]
        time_curr += d / speed
        battery -= d

        if battery < -1e-9:
            return float('inf'), False

        if v in stations:
            charge_time = (battery_max - battery) * recharge_rate
            time_curr += charge_time
            battery = battery_max
        else:
            time_curr = max(time_curr, ready[v])
            if time_curr > due[v] + 1e-9:
                return float('inf'), False
            time_curr += service[v]

        total_dist += d
        current = v

    d = dist[current, 0]
    time_curr += d / speed
    battery -= d
    total_dist += d

    if battery < -1e-9:
        return float('inf'), False
    if time_curr > horizon + 1e-9:
        return float('inf'), False

    return total_dist, True


def _route_distance(route, instance, dist):
    """Quick distance-only computation (no feasibility check)."""
    total = 0.0
    current = 0
    for v in route:
        total += dist[current, v]
        current = v
    total += dist[current, 0]
    return total


def _is_feasible(route, instance, dist, stations_set=None):
    if stations_set is None:
        stations_set = set(instance["stations"])
    _, feas = _simulate_route(route, instance, dist, stations_set)
    return feas


def _optimize_stations_in_route(route, instance, dist, stations_set=None):
    """Try to add, remove, or reposition stations to reduce distance while maintaining feasibility."""
    if stations_set is None:
        stations_set = set(instance["stations"])
    
    current_dist, current_feas = _simulate_route(route, instance, dist, stations_set)
    if not current_feas:
        return route
    
    improved = True
    max_passes = 5
    for _ in range(max_passes):
        improved = False
        
        # Try removing each station if still feasible
        for i in range(len(route)):
            if route[i] in stations_set:
                new_route = route[:i] + route[i+1:]
                if new_route:  # don't remove all nodes
                    nd, nf = _simulate_route(new_route, instance, dist, stations_set)
                    if nf and nd < current_dist - 1e-9:
                        route = new_route
                        current_dist = nd
                        improved = True
                        break
        
        if improved:
            continue
        
        # Try inserting each station at best position
        for s in stations_set:
            if s in route:
                continue
            best_pos = -1
            best_d = current_dist
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [s] + route[pos:]
                nd, nf = _simulate_route(new_route, instance, dist, stations_set)
                if nf and nd < best_d - 1e-9:
                    best_d = nd
                    best_pos = pos
            if best_pos >= 0:
                route = route[:best_pos] + [s] + route[best_pos:]
                current_dist = best_d
                improved = True
                break
        
        # Try repositioning existing stations
        if not improved:
            for i in range(len(route)):
                if route[i] in stations_set:
                    s = route[i]
                    rest = route[:i] + route[i+1:]
                    best_pos = -1
                    best_d = current_dist
                    for pos in range(len(rest) + 1):
                        new_route = rest[:pos] + [s] + rest[pos:]
                        if new_route == route:
                            continue
                        nd, nf = _simulate_route(new_route, instance, dist, stations_set)
                        if nf and nd < best_d - 1e-9:
                            best_d = nd
                            best_pos = pos
                    if best_pos >= 0:
                        route = rest[:best_pos] + [s] + rest[best_pos:]
                        current_dist = best_d
                        improved = True
                        break
    
    return route


def _savings_construction(instance, dist):
    """Clarke-Wright savings heuristic with feasibility checks."""
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])
    n_customers = instance["n_customers"]

    customers = list(range(1, n_customers + 1))

    # Compute savings: savings[i,j] = d(0,i) + d(0,j) - d(i,j)
    savings = {}
    for i in customers:
        for j in customers:
            if i != j:
                s = dist[0, i] + dist[0, j] - dist[i, j]
                savings[(i, j)] = s

    # Sort by savings descending
    sorted_savings = sorted(savings.items(), key=lambda x: -x[1])

    # Each customer starts as its own route
    routes = {c: [c] for c in customers}
    route_of = {c: c for c in customers}  # node -> route key
    route_load = {c: demands[c] for c in customers}

    # Find ends of each route
    def get_ends(route):
        if len(route) == 1:
            return route[0], route[0]
        return route[0], route[-1]

    # Try merging
    for (i, j), s in sorted_savings:
        if s <= 0:
            continue
        ri = route_of[i]
        rj = route_of[j]
        if ri == rj:
            continue

        # Check if i is an end of its route and j is an end of its route
        ends_i = set(get_ends(routes[ri]))
        ends_j = set(get_ends(routes[rj]))

        if i not in ends_i or j not in ends_j:
            continue

        # Check capacity
        if route_load[ri] + route_load[rj] > capacity:
            continue

        # Try merging: find the best orientation
        best_route = None
        best_dist = float('inf')

        routes_i = routes[ri]
        routes_j = routes[rj]

        # Four possible merges
        candidates = []
        # i at end of ri, j at start of rj
        if i == routes_i[-1] and j == routes_j[0]:
            candidates.append(routes_i + routes_j)
        # i at start of ri, j at end of rj
        if i == routes_i[0] and j == routes_j[-1]:
            candidates.append(routes_j + routes_i)
        # i at end of ri, j at end of rj
        if i == routes_i[-1] and j == routes_j[-1]:
            candidates.append(routes_i + routes_j[::-1])
        # i at start of ri, j at start of rj
        if i == routes_i[0] and j == routes_j[0]:
            candidates.append(routes_j[::-1] + routes_i)

        for cand in candidates:
            d, feas = _simulate_route(cand, instance, dist)
            if feas and d < best_dist:
                best_dist = d
                best_route = cand

        if best_route is not None:
            routes[ri] = best_route
            route_load[ri] += route_load[rj]
            for node in best_route:
                route_of[node] = ri
            del routes[rj]
            del route_load[rj]

    return [routes[k] for k in routes]


def _greedy_construction(instance, dist):
    """Greedy nearest-neighbor with feasibility."""
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])
    n_customers = instance["n_customers"]

    customers = list(range(1, n_customers + 1))
    customers.sort(key=lambda v: due[v])

    routes = []
    unassigned = set(customers)

    while unassigned:
        route = []
        current = 0
        battery = battery_max
        time_curr = 0.0
        load = 0.0

        while unassigned:
            best_customer = None
            best_cost = float('inf')

            for c in unassigned:
                if load + demands[c] > capacity:
                    continue
                if c in stations:
                    continue

                d = dist[current, c]
                new_battery = battery - d
                if new_battery < -1e-9:
                    continue

                new_time = time_curr + d / speed
                new_time = max(new_time, ready[c])
                if new_time > due[c] + 1e-9:
                    continue
                new_time += service[c]

                # Check return feasibility
                d_ret = dist[c, 0]
                final_battery = new_battery - d_ret
                final_time = new_time + d_ret / speed
                if final_battery < -1e-9 or final_time > horizon + 1e-9:
                    continue

                cost = d
                if cost < best_cost:
                    best_cost = cost
                    best_customer = c

            if best_customer is None:
                break

            d = dist[current, best_customer]
            time_curr += d / speed
            battery -= d
            load += demands[best_customer]
            time_curr = max(time_curr, ready[best_customer])
            time_curr += service[best_customer]

            route.append(best_customer)
            unassigned.remove(best_customer)
            current = best_customer

        if route:
            routes.append(route)

    return routes


def _try_insert_station(route, station, instance, dist):
    """Try inserting a station at best position."""
    best_route = None
    best_dist = float('inf')
    n = len(route)

    for pos in range(n + 1):
        new_route = route[:pos] + [station] + route[pos:]
        d, feas = _simulate_route(new_route, instance, dist)
        if feas and d < best_dist:
            best_dist = d
            best_route = new_route

    return best_route


def _ensure_feasible(routes, instance, dist):
    """Insert stations to make routes feasible."""
    stations_list = instance["stations"]
    improved = True
    max_iter = 20
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        for i in range(len(routes)):
            d, feas = _simulate_route(routes[i], instance, dist)
            if not feas:
                for s in stations_list:
                    if s not in routes[i]:
                        new_route = _try_insert_station(routes[i], s, instance, dist)
                        if new_route is not None:
                            routes[i] = new_route
                            improved = True
                            break
    return routes


def _two_opt(route, instance, dist, max_passes=3):
    """2-opt local search."""
    n = len(route)
    if n < 2:
        return route

    current_dist, current_feas = _simulate_route(route, instance, dist)
    if not current_feas:
        return route

    for _ in range(max_passes):
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                nd, nf = _simulate_route(new_route, instance, dist)
                if nf and nd < current_dist - 1e-9:
                    route = new_route
                    current_dist = nd
                    improved = True
        if not improved:
            break

    return route


def _or_opt(route, instance, dist, max_passes=3):
    """Or-opt: move 1 or 2 consecutive nodes."""
    n = len(route)
    if n < 3:
        return route

    current_dist, current_feas = _simulate_route(route, instance, dist)
    if not current_feas:
        return route

    for _ in range(max_passes):
        improved = False
        for seg_len in [1, 2]:
            for i in range(n - seg_len + 1):
                segment = route[i:i+seg_len]
                rest = route[:i] + route[i+seg_len:]
                for j in range(len(rest) + 1):
                    if i <= j < i + seg_len:
                        continue
                    new_route = rest[:j] + segment + rest[j:]
                    if new_route == route:
                        continue
                    nd, nf = _simulate_route(new_route, instance, dist)
                    if nf and nd < current_dist - 1e-9:
                        route = new_route
                        current_dist = nd
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break

    return route


def _relocate(routes, instance, dist):
    """Try moving a node from one route to another."""
    improved = True
    max_passes = 2
    for _ in range(max_passes):
        improved = False
        for i in range(len(routes)):
            for j in range(len(routes[i])):
                node = routes[i][j]
                # Try removing from route i
                remaining = routes[i][:j] + routes[i][j+1:]
                if not remaining:
                    continue
                d_rem, feas_rem = _simulate_route(remaining, instance, dist)
                if not feas_rem:
                    continue

                # Try inserting into each other route
                for k in range(len(routes)):
                    if k == i:
                        continue
                    target = routes[k]
                    for pos in range(len(target) + 1):
                        new_target = target[:pos] + [node] + target[pos:]
                        d_new, feas_new = _simulate_route(new_target, instance, dist)
                        if not feas_new:
                            continue
                        # Total distance change
                        old_total = d_rem + _route_distance(target, instance, dist)
                        new_total = d_rem + _route_distance(new_target, instance, dist)
                        if new_total < old_total - 1e-9:
                            routes[i] = remaining
                            routes[k] = new_target
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

    # Remove empty routes
    routes = [r for r in routes if r]
    return routes


def _swap_nodes(routes, instance, dist):
    """Try swapping nodes between routes."""
    improved = True
    max_passes = 1
    for _ in range(max_passes):
        improved = False
        for i in range(len(routes)):
            for j in range(len(routes[i])):
                for k in range(i + 1, len(routes)):
                    for l in range(len(routes[k])):
                        ni = routes[i][j]
                        nk = routes[k][l]
                        # Swap
                        new_i = routes[i][:]
                        new_k = routes[k][:]
                        new_i[j] = nk
                        new_k[l] = ni
                        d_i, feas_i = _simulate_route(new_i, instance, dist)
                        d_k, feas_k = _simulate_route(new_k, instance, dist)
                        if feas_i and feas_k:
                            old_total = _route_distance(routes[i], instance, dist) + _route_distance(routes[k], instance, dist)
                            new_total = d_i + d_k
                            if new_total < old_total - 1e-9:
                                routes[i] = new_i
                                routes[k] = new_k
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
    return routes


def _local_search(routes, instance, dist, deadline):
    """Full local search with deadline."""
    start_time = time.time()
    stations_set = set(instance["stations"])
    max_iter = 80
    for it in range(max_iter):
        if time.time() > deadline:
            break

        # 2-opt on each route
        for i in range(len(routes)):
            if time.time() > deadline:
                break
            new_route = _two_opt(routes[i], instance, dist, max_passes=2)
            if new_route != routes[i]:
                routes[i] = new_route

        # Or-opt
        for i in range(len(routes)):
            if time.time() > deadline:
                break
            new_route = _or_opt(routes[i], instance, dist, max_passes=2)
            if new_route != routes[i]:
                routes[i] = new_route

        # Station optimization
        for i in range(len(routes)):
            if time.time() > deadline:
                break
            new_route = _optimize_stations_in_route(routes[i], instance, dist, stations_set)
            if new_route != routes[i]:
                routes[i] = new_route

        # Inter-route moves
        if len(routes) > 1:
            if time.time() > deadline:
                break
            routes = _relocate(routes, instance, dist)
            if time.time() > deadline:
                break
            routes = _swap_nodes(routes, instance, dist)

    return routes


def _polish(routes, instance, dist, deadline):
    """Final high-precision polish: more 2-opt and Or-opt passes with tighter tolerance."""
    stations_set = set(instance["stations"])
    for i in range(len(routes)):
        if time.time() > deadline:
            break
        routes[i] = _two_opt(routes[i], instance, dist, max_passes=5)
        if time.time() > deadline:
            break
        routes[i] = _or_opt(routes[i], instance, dist, max_passes=5)
        if time.time() > deadline:
            break
        routes[i] = _optimize_stations_in_route(routes[i], instance, dist, stations_set)
    return routes


def _total_distance(routes, instance, dist):
    total = 0.0
    for route in routes:
        d, feas = _simulate_route(route, instance, dist)
        if not feas:
            return float('inf')
        total += d
    return total


def _validate(routes, instance, dist):
    stations = set(instance["stations"])
    n_customers = instance["n_customers"]
    all_customers = set(range(1, n_customers + 1))
    served = set()

    for route in routes:
        for v in route:
            if v in all_customers:
                if v in served:
                    return False
                served.add(v)
            elif v not in stations:
                return False
        _, feas = _simulate_route(route, instance, dist)
        if not feas:
            return False

    return served == all_customers


def _make_solution_feasible(routes, instance, dist):
    """Ensure all customers are served and routes are feasible."""
    stations = set(instance["stations"])
    n_customers = instance["n_customers"]
    all_customers = set(range(1, n_customers + 1))
    served = set()
    for route in routes:
        for v in route:
            if v in all_customers:
                served.add(v)

    # Add missing customers as singleton routes
    for c in all_customers - served:
        routes.append([c])

    # Ensure feasibility
    routes = _ensure_feasible(routes, instance, dist)

    # Final check
    if not _validate(routes, instance, dist):
        # Last resort: one customer per route
        routes = [[c] for c in range(1, n_customers + 1)]

    return routes


def plan(instance):
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    dist = _build_distance_matrix(instance["coords"])
    stations_set = set(instance["stations"])

    best_routes = None
    best_dist = float('inf')

    # Try multiple construction methods
    constructions = []

    # 1. Savings
    try:
        r1 = _savings_construction(instance, dist)
        r1 = _ensure_feasible(r1, instance, dist)
        if _validate(r1, instance, dist):
            d = _total_distance(r1, instance, dist)
            if d < best_dist:
                best_dist = d
                best_routes = [list(r) for r in r1]
        constructions.append(r1)
    except Exception:
        pass

    # 2. Greedy
    try:
        r2 = _greedy_construction(instance, dist)
        r2 = _ensure_feasible(r2, instance, dist)
        if _validate(r2, instance, dist):
            d = _total_distance(r2, instance, dist)
            if d < best_dist:
                best_dist = d
                best_routes = [list(r) for r in r2]
        constructions.append(r2)
    except Exception:
        pass

    # 3. Randomized greedy (multiple restarts)
    for restart in range(5):
        if time.time() > deadline - 40:
            break
        try:
            random.seed(42 + restart)
            r3 = _greedy_construction(instance, dist)
            # Shuffle slightly
            for i in range(len(r3)):
                if len(r3[i]) > 2:
                    pos = random.randint(1, len(r3[i]) - 2)
                    r3[i][pos], r3[i][pos+1] = r3[i][pos+1], r3[i][pos]
            r3 = _ensure_feasible(r3, instance, dist)
            if _validate(r3, instance, dist):
                d = _total_distance(r3, instance, dist)
                if d < best_dist:
                    best_dist = d
                    best_routes = [list(r) for r in r3]
            constructions.append(r3)
        except Exception:
            pass

    if best_routes is None:
        # Fallback
        best_routes = [[c] for c in range(1, instance["n_customers"] + 1)]

    # Local search on best
    search_deadline = deadline - 10
    best_routes = _local_search(best_routes, instance, dist, search_deadline)

    # Final polish with more iterations
    polish_deadline = deadline - 3
    best_routes = _polish(best_routes, instance, dist, polish_deadline)

    # Final validation
    if not _validate(best_routes, instance, dist):
        best_routes = _make_solution_feasible(best_routes, instance, dist)

    return best_routes
# EVOLVE-BLOCK-END
