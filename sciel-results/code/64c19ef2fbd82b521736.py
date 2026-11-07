"""EV routing with time windows: Clarke-Wright + LNS + Or-opt + inter-route 2-opt."""
# EVOLVE-BLOCK-START
import numpy as np
import math
import time
import random

_DEADLINE = None
_MARGIN = 15.0

def _check_deadline():
    if _DEADLINE is not None and time.time() > _DEADLINE - _MARGIN:
        return True
    return False

def _build_dist(coords):
    n = len(coords)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            v = math.hypot(dx, dy)
            d[i][j] = v
            d[j][i] = v
    return d

def _route_distance(route, dist):
    if not route:
        return 0.0
    total = dist[0, route[0]]
    for i in range(len(route) - 1):
        total += dist[route[i], route[i + 1]]
    total += dist[route[-1], 0]
    return total

def _simulate_route(route, dist, ready, due, service, demands, station_set,
                    battery_max, recharge_rate, horizon, capacity, speed=1.0):
    """Returns (feasible, total_distance)."""
    total_demand = 0
    for v in route:
        if v not in station_set:
            total_demand += demands[v]
    if total_demand > capacity:
        return (False, float('inf'))
    t = 0.0
    battery = float(battery_max)
    total_dist = 0.0
    prev = 0
    for v in route:
        d = dist[prev, v]
        t += d
        battery -= d
        total_dist += d
        if battery < -1e-9:
            return (False, float('inf'))
        if v in station_set:
            charge_time = (battery_max - battery) * recharge_rate
            t += charge_time
            battery = float(battery_max)
        else:
            if t < ready[v]:
                t = ready[v]
            if t > due[v] + 1e-9:
                return (False, float('inf'))
            t += service[v]
        prev = v
    d = dist[prev, 0]
    t += d
    battery -= d
    total_dist += d
    if battery < -1e-9:
        return (False, float('inf'))
    if t > horizon + 1e-9:
        return (False, float('inf'))
    return (True, total_dist)

def _best_insertion(cust, route, dist, ready, due, service, demands, station_set,
                    battery_max, recharge_rate, horizon, capacity, speed=1.0):
    """Find best position to insert cust into route, or None if infeasible."""
    best_pos = -1
    best_d = float('inf')
    for pos in range(len(route) + 1):
        new_route = route[:pos] + [cust] + route[pos:]
        f, d = _simulate_route(new_route, dist, ready, due, service, demands,
                               station_set, battery_max, recharge_rate, horizon,
                               capacity, speed)
        if f and d < best_d:
            best_d = d
            best_pos = pos
    return best_pos, best_d

def _add_charging_if_needed(route, dist, ready, due, service, demands, station_set,
                            battery_max, recharge_rate, horizon, capacity, speed=1.0):
    """If route is infeasible due to battery, try inserting a station."""
    f, d = _simulate_route(route, dist, ready, due, service, demands,
                           station_set, battery_max, recharge_rate, horizon,
                           capacity, speed)
    if f:
        return route
    # Try inserting each station at each position
    stations = list(station_set)
    for s in stations:
        for pos in range(len(route) + 1):
            new_route = route[:pos] + [s] + route[pos:]
            f2, d2 = _simulate_route(new_route, dist, ready, due, service, demands,
                                     station_set, battery_max, recharge_rate, horizon,
                                     capacity, speed)
            if f2:
                return new_route
    return None

def _clarke_wright_build(instance, dist):
    """Build initial solution using Clarke-Wright savings."""
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = instance["stations"]
    station_set = set(stations)
    battery_max = instance["battery"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    capacity = instance["capacity"]
    speed = instance.get("speed", 1.0)

    # Compute savings
    savings = []
    for i in customers:
        for j in customers:
            if i < j:
                s = dist[0, i] + dist[0, j] - dist[i, j]
                savings.append((s, i, j))
    savings.sort(reverse=True)

    # Start with each customer in its own route
    routes = {c: [c] for c in customers}
    parent = {c: c for c in customers}
    loads = {c: demands[c] for c in customers}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for s, i, j in savings:
        if _check_deadline():
            break
        if s <= 0:
            break
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        if loads[ri] + loads[rj] > capacity:
            continue
        route_i = routes[ri]
        route_j = routes[rj]
        # Try to merge: i should be at an end of route_i, j at an end of route_j
        # Try all 4 combinations
        merged = None
        for ri_end in [route_i, route_i[::-1]]:
            for rj_end in [route_j, route_j[::-1]]:
                if ri_end[-1] == i and rj_end[0] == j:
                    candidate = ri_end + rj_end[1:]
                    f, d = _simulate_route(candidate, dist, ready, due, service,
                                           demands, station_set, battery_max,
                                           recharge_rate, horizon, capacity, speed)
                    if f:
                        merged = candidate
                        break
                if ri_end[-1] == j and rj_end[0] == i:
                    candidate = ri_end + rj_end[1:]
                    f, d = _simulate_route(candidate, dist, ready, due, service,
                                           demands, station_set, battery_max,
                                           recharge_rate, horizon, capacity, speed)
                    if f:
                        merged = candidate
                        break
            if merged:
                break
        if merged:
            union(ri, rj)
            routes[find(i)] = merged
            loads[find(i)] = loads[ri] + loads[rj]

    return [routes[find(c)] for c in customers if find(c) == c]

def _greedy_build(instance, dist):
    """Greedy nearest-neighbor with time windows."""
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = instance["stations"]
    station_set = set(stations)
    battery_max = instance["battery"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    capacity = instance["capacity"]
    speed = instance.get("speed", 1.0)

    # Sort by due time
    customers_sorted = sorted(customers, key=lambda v: (due[v], ready[v]))
    routes = []
    route_loads = []
    unassigned = set(customers_sorted)

    while unassigned:
        if _check_deadline():
            break
        # Start a new route with the earliest-due unassigned customer
        start = min(unassigned, key=lambda v: (due[v], ready[v]))
        route = [start]
        load = demands[start]
        unassigned.discard(start)
        prev = start
        t = dist[0, start]
        battery = battery_max - dist[0, start]
        if t < ready[start]:
            t = ready[start]
        t += service[start]

        while unassigned:
            if _check_deadline():
                break
            best_cust = None
            best_dist = float('inf')
            for c in unassigned:
                if load + demands[c] > capacity:
                    continue
                d = dist[prev, c]
                if d >= best_dist:
                    continue
                new_t = t + d
                new_batt = battery - d
                if new_batt < -1e-9:
                    continue
                if c in station_set:
                    new_t += (battery_max - new_batt) * recharge_rate
                    new_batt = battery_max
                else:
                    if new_t < ready[c]:
                        new_t = ready[c]
                    if new_t > due[c] + 1e-9:
                        continue
                    new_t += service[c]
                # Check return to depot
                ret_d = dist[c, 0]
                if new_batt - ret_d < -1e-9:
                    continue
                if new_t + ret_d > horizon + 1e-9:
                    continue
                best_cust = c
                best_dist = d
            if best_cust is None:
                break
            d = dist[prev, best_cust]
            t += d
            battery -= d
            if best_cust in station_set:
                t += (battery_max - battery) * recharge_rate
                battery = battery_max
            else:
                if t < ready[best_cust]:
                    t = ready[best_cust]
                t += service[best_cust]
                load += demands[best_cust]
            route.append(best_cust)
            prev = best_cust
            unassigned.discard(best_cust)

        # Verify route feasibility
        f, _ = _simulate_route(route, dist, ready, due, service, demands,
                               station_set, battery_max, recharge_rate, horizon,
                               capacity, speed)
        if f:
            routes.append(route)
            route_loads.append(load)

    return routes

def _solve_instance(instance):
    coords = instance["coords"]
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = instance["stations"]
    station_set = set(stations)
    battery_max = instance["battery"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    capacity = instance["capacity"]
    speed = instance.get("speed", 1.0)

    dist = _build_dist(coords)
    n_nodes = len(coords)

    # Try Clarke-Wright first, fall back to greedy
    routes = _clarke_wright_build(instance, dist)
    # Verify all customers served
    served = set()
    for r in routes:
        for v in r:
            if v not in station_set:
                served.add(v)
    if served != set(customers):
        routes = _greedy_build(instance, dist)
        served = set()
        for r in routes:
            for v in r:
                if v not in station_set:
                    served.add(v)
        if served != set(customers):
            # Last resort: one customer per route
            routes = [[c] for c in customers]

    # Local search
    best_routes = routes
    best_total = sum(_route_distance(r, dist) for r in best_routes)
    no_improve = 0
    max_no_improve = 15
    rng = random.Random(42)

    while not _check_deadline():
        improved = False

        # 2-opt within routes
        for r in range(len(routes)):
            if len(routes[r]) < 3:
                continue
            route = routes[r]
            for i in range(len(route) - 1):
                for j in range(i + 1, len(route)):
                    if j - i == 1:
                        continue
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    f, d = _simulate_route(new_route, dist, ready, due, service,
                                           demands, station_set, battery_max,
                                           recharge_rate, horizon, capacity, speed)
                    if f:
                        old_d = _route_distance(route, dist)
                        if d < old_d - 1e-9:
                            routes[r] = new_route
                            best_total = sum(_route_distance(rr, dist) for rr in routes)
                            improved = True

        # Or-opt: move 1-3 consecutive nodes within a route
        for r in range(len(routes)):
            route = routes[r]
            for seg_len in [1, 2, 3]:
                for i in range(len(route) - seg_len + 1):
                    segment = route[i:i+seg_len]
                    rest = route[:i] + route[i+seg_len:]
                    for j in range(len(rest) + 1):
                        new_route = rest[:j] + segment + rest[j:]
                        if new_route == route:
                            continue
                        f, d = _simulate_route(new_route, dist, ready, due, service,
                                               demands, station_set, battery_max,
                                               recharge_rate, horizon, capacity, speed)
                        if f:
                            old_d = _route_distance(route, dist)
                            if d < old_d - 1e-9:
                                routes[r] = new_route
                                route = new_route
                                best_total = sum(_route_distance(rr, dist) for rr in routes)
                                improved = True

        # Relocate between routes
        for r1 in range(len(routes)):
            if not routes[r1]:
                continue
            for i in range(len(routes[r1])):
                cust = routes[r1][i]
                if cust in station_set:
                    continue
                new_r1 = routes[r1][:i] + routes[r1][i+1:]
                for r2 in range(len(routes)):
                    if r1 == r2:
                        continue
                    for j in range(len(routes[r2]) + 1):
                        new_r2 = routes[r2][:j] + [cust] + routes[r2][j:]
                        f1, d1 = _simulate_route(new_r1, dist, ready, due, service,
                                                 demands, station_set, battery_max,
                                                 recharge_rate, horizon, capacity, speed)
                        f2, d2 = _simulate_route(new_r2, dist, ready, due, service,
                                                 demands, station_set, battery_max,
                                                 recharge_rate, horizon, capacity, speed)
                        if f1 and f2:
                            old_total = _route_distance(routes[r1], dist) + _route_distance(routes[r2], dist)
                            new_total = d1 + d2
                            if new_total < old_total - 1e-9:
                                routes[r1] = new_r1
                                routes[r2] = new_r2
                                best_total = sum(_route_distance(rr, dist) for rr in routes)
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

        # Swap between routes
        for r1 in range(len(routes)):
            if len(routes[r1]) < 1:
                continue
            for i in range(len(routes[r1])):
                c1 = routes[r1][i]
                if c1 in station_set:
                    continue
                for r2 in range(len(routes)):
                    if r1 == r2:
                        continue
                    for j in range(len(routes[r2])):
                        c2 = routes[r2][j]
                        if c2 in station_set or c1 == c2:
                            continue
                        new_r1 = routes[r1][:]
                        new_r2 = routes[r2][:]
                        new_r1[i] = c2
                        new_r2[j] = c1
                        f1, d1 = _simulate_route(new_r1, dist, ready, due, service,
                                                 demands, station_set, battery_max,
                                                 recharge_rate, horizon, capacity, speed)
                        f2, d2 = _simulate_route(new_r2, dist, ready, due, service,
                                                 demands, station_set, battery_max,
                                                 recharge_rate, horizon, capacity, speed)
                        if f1 and f2:
                            old_total = _route_distance(routes[r1], dist) + _route_distance(routes[r2], dist)
                            new_total = d1 + d2
                            if new_total < old_total - 1e-9:
                                routes[r1] = new_r1
                                routes[r2] = new_r2
                                best_total = sum(_route_distance(rr, dist) for rr in routes)
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

        # Inter-route 2-opt (cross)
        for r1 in range(len(routes)):
            if len(routes[r1]) < 2:
                continue
            for r2 in range(r1 + 1, len(routes)):
                if len(routes[r2]) < 2:
                    continue
                route1 = routes[r1]
                route2 = routes[r2]
                for i in range(len(route1) - 1):
                    for j in range(len(route2) - 1):
                        # Swap the tails
                        new_r1 = route1[:i+1] + route2[j+1:][::-1]
                        new_r2 = route2[:j+1] + route1[i+1:][::-1]
                        f1, d1 = _simulate_route(new_r1, dist, ready, due, service,
                                                 demands, station_set, battery_max,
                                                 recharge_rate, horizon, capacity, speed)
                        f2, d2 = _simulate_route(new_r2, dist, ready, due, service,
                                                 demands, station_set, battery_max,
                                                 recharge_rate, horizon, capacity, speed)
                        if f1 and f2:
                            old_total = _route_distance(route1, dist) + _route_distance(route2, dist)
                            new_total = d1 + d2
                            if new_total < old_total - 1e-9:
                                routes[r1] = new_r1
                                routes[r2] = new_r2
                                best_total = sum(_route_distance(rr, dist) for rr in routes)
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

        # LNS: remove k customers and reinsert
        if not improved:
            no_improve += 1
            if no_improve >= max_no_improve:
                break
            # LNS destroy: remove random subset
            all_custs = []
            for r in routes:
                for v in r:
                    if v not in station_set:
                        all_custs.append((r, v))
            if len(all_custs) < 3:
                continue
            k = min(rng.randint(3, 7), len(all_custs))
            removed = rng.sample(all_custs, k)
            removed_set = set(v for _, v in removed)
            # Remove from routes
            for r_idx in range(len(routes)):
                routes[r_idx] = [v for v in routes[r_idx] if v not in removed_set]
            routes = [r for r in routes if r]
            # Reinsert one by one
            for _, cust in removed:
                best_r = -1
                best_pos = -1
                best_d = float('inf')
                for r_idx in range(len(routes)):
                    pos, d = _best_insertion(cust, routes[r_idx], dist, ready, due,
                                             service, demands, station_set, battery_max,
                                             recharge_rate, horizon, capacity, speed)
                    if pos >= 0 and d < best_d:
                        best_d = d
                        best_r = r_idx
                        best_pos = pos
                if best_r >= 0:
                    routes[best_r].insert(best_pos, cust)
                else:
                    # Try with charging
                    for r_idx in range(len(routes)):
                        pos, d = _best_insertion(cust, routes[r_idx], dist, ready, due,
                                                 service, demands, station_set, battery_max,
                                                 recharge_rate, horizon, capacity, speed)
                        if pos >= 0:
                            routes[best_r].insert(best_pos, cust)
                            break
                    else:
                        # Put in a new route
                        routes.append([cust])
                improved = True
            no_improve = 0

        if improved:
            no_improve = 0
        else:
            no_improve += 1

    # Final cleanup
    routes = [r for r in routes if r]

    # Verify
    served = set()
    for r in routes:
        for v in r:
            if v not in station_set:
                served.add(v)
    if served != set(customers):
        routes = [[c] for c in customers]

    return routes

def plan(instance):
    global _DEADLINE
    _DEADLINE = time.time() + 150.0 - _MARGIN

    routes = _solve_instance(instance)

    # Verify all customers served
    served = set()
    station_set = set(instance["stations"])
    for r in routes:
        for v in r:
            if v not in station_set:
                served.add(v)
    expected = set(range(1, instance["n_customers"] + 1))
    if served != expected:
        routes = [[v] for v in range(1, instance["n_customers"] + 1)]

    return routes
# EVOLVE-BLOCK-END
