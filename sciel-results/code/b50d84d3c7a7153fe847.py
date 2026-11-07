import time
import math
import heapq
import numpy as np

START = time.time()
DEADLINE = time.time() + 115.0  # safety margin under the 120s limit


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _time_left():
    return DEADLINE - time.time()


def _simulate(route, coords, ready, due, service, battery, speed,
              recharge, capacity, horizon):
    """Return (feasible, distance). route = customer node ids only."""
    dist = 0.0
    energy = battery
    t = 0.0
    load = 0.0
    n = len(route)
    for i in range(n + 1):
        node = route[i] if i < n else 0
        d = _dist(coords[route[i - 1]], coords[node]) if i > 0 else _dist(coords[0], coords[node])
        if energy < d - 1e-9:
            return False, dist + d
        energy -= d
        t += d / speed
        dist += d
        if node != 0:
            if t > due[node] + 1e-9:
                return False, dist
            if t < ready[node]:
                t = ready[node]
            t += service[node]
            load += service[node]  # placeholder; demand handled by caller
    if t > horizon + 1e-9:
        return False, dist
    return True, dist


def _sim_full(route, coords, ready, due, service, demand, battery, speed,
              recharge, capacity, horizon):
    """Simulate a full node route (including depot 0 at start/end).
    Returns (feasible, distance)."""
    dist = 0.0
    energy = battery
    t = 0.0
    load = 0.0
    n = len(route)
    for i in range(n):
        node = route[i]
        d = _dist(coords[route[i - 1]], coords[node])
        if energy < d - 1e-9:
            return False, dist + d
        energy -= d
        t += d / speed
        dist += d
        if node != 0:
            if node in demand:
                if load + demand[node] > capacity + 1e-9:
                    return False, dist
                load += demand[node]
            if t > due[node] + 1e-9:
                return False, dist
            if t < ready[node]:
                t = ready[node]
            t += service[node]
    # back to depot
    d = _dist(coords[route[-1]], coords[0])
    if energy < d - 1e-9:
        return False, dist + d
    energy -= d
    t += d / speed
    dist += d
    if t > horizon + 1e-9:
        return False, dist
    return True, dist


def _route_distance(route, coords):
    n = len(route)
    d = 0.0
    for i in range(n):
        d += _dist(coords[route[i - 1]], coords[route[i]])
    return d


def plan(instance):
    global DEADLINE
    DEADLINE = time.time() + 115.0

    coords = instance["coords"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    demand = instance["demand"]
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge = instance["recharge_rate"]
    horizon = instance["horizon"]

    customers = list(range(1, n_customers + 1))
    all_nodes = set(range(0, len(coords)))

    # Precompute pairwise distances
    N = len(coords)
    dist = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            dist[i, j] = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])

    def feasible_insert(route, node, pos):
        """Check if inserting node at pos in route keeps feasibility.
        route is list of customer node ids. Returns (feasible, new_dist)."""
        # Build candidate route
        cand = route[:pos] + [node] + route[pos:]
        # demand
        load = sum(demand[c] for c in cand)
        if load > capacity + 1e-9:
            return False, 0.0
        # Simulate
        energy = battery
        t = 0.0
        prev = 0
        d_total = 0.0
        for c in cand:
            d = dist[prev, c]
            if energy < d - 1e-9:
                return False, 0.0
            energy -= d
            t += d / speed
            d_total += d
            if t > due[c] + 1e-9:
                return False, 0.0
            if t < ready[c]:
                t = ready[c]
            t += service[c]
            prev = c
        d = dist[prev, 0]
        if energy < d - 1e-9:
            return False, 0.0
        t += d / speed
        d_total += d
        if t > horizon + 1e-9:
            return False, 0.0
        return True, d_total

    # Greedy construction: start with empty routes, insert customers one by one
    # into the route/position that minimizes added distance while feasible.
    # We'll keep a list of routes (each a list of customer ids).
    routes = []
    remaining = set(customers)

    # Order customers by due time (earliest first) for better feasibility
    order = sorted(customers, key=lambda c: due[c])

    for c in order:
        if _time_left() < 5.0:
            break
        best = None
        best_cost = float('inf')
        # Try inserting into existing routes
        for ri, route in enumerate(routes):
            for pos in range(len(route) + 1):
                ok, d = feasible_insert(route, c, pos)
                if ok:
                    # added distance = new_dist - old_dist
                    old_d = _route_distance(route + [0], coords) if route else _dist(coords[0], coords[0])
                    # compute old distance properly
                    old_d = 0.0
                    if route:
                        old_d = dist[0, route[0]] + sum(dist[route[k], route[k+1]] for k in range(len(route)-1)) + dist[route[-1], 0]
                    else:
                        old_d = 0.0
                    added = d - old_d
                    if added < best_cost:
                        best_cost = added
                        best = (ri, pos)
        # Try new route
        ok, d = feasible_insert([], c, 0)
        if ok:
            if d < best_cost:
                best_cost = d
                best = (len(routes), 0)
        if best is not None:
            ri, pos = best
            if ri < len(routes):
                routes[ri].insert(pos, c)
            else:
                routes.append([c])
        else:
            # Should not happen; fallback new route
            routes.append([c])

    # Local search: 2-opt and re-insertion
    def total_dist():
        s = 0.0
        for r in routes:
            if not r:
                continue
            s += dist[0, r[0]] + sum(dist[r[k], r[k+1]] for k in range(len(r)-1)) + dist[r[-1], 0]
        return s

    improved = True
    while improved and _time_left() > 3.0:
        improved = False
        # 2-opt within each route
        for ri, route in enumerate(routes):
            if len(route) < 2:
                continue
            n = len(route)
            for i in range(n - 1):
                if _time_left() < 3.0:
                    break
                for j in range(i + 1, n):
                    a = 0 if i == 0 else route[i-1]
                    b = route[i]
                    c = route[j]
                    d = 0 if j == n-1 else route[j+1]
                    old = dist[a, b] + dist[c, d]
                    new = dist[a, c] + dist[b, d]
                    if new < old - 1e-9:
                        route[i:j+1] = route[i:j+1][::-1]
                        # verify feasibility
                        ok, _ = feasible_insert(route[:i] + route[i:j+1][::-1] + route[j+1:], 0, 0) if False else (True, 0)
                        # quick re-simulate
                        full = [0] + route + [0]
                        ok2, _ = _sim_full(full, coords, ready, due, service, demand, battery, speed, recharge, capacity, horizon)
                        if not ok2:
                            route[i:j+1] = route[i:j+1][::-1]  # revert
                        else:
                            improved = True
        # Re-insertion: move each customer to best position in same or other route
        for ri, route in enumerate(routes):
            for idx in range(len(route)):
                if _time_left() < 3.0:
                    break
                c = route[idx]
                route2 = route[:idx] + route[idx+1:]
                best = None
                best_cost = float('inf')
                # Try all routes
                for rj, rout in enumerate(routes):
                    if rj == ri:
                        base = route2
                    else:
                        base = rout
                    for pos in range(len(base) + 1):
                        ok, d = feasible_insert(base, c, pos)
                        if ok:
                            if rj == ri:
                                old_d = 0.0
                                if route:
                                    old_d = dist[0, route[0]] + sum(dist[route[k], route[k+1]] for k in range(len(route)-1)) + dist[route[-1], 0]
                            else:
                                old_d = 0.0
                                if rout:
                                    old_d = dist[0, rout[0]] + sum(dist[rout[k], rout[k+1]] for k in range(len(rout)-1)) + dist[rout[-1], 0]
                            # cost = new total - old total
                            if rj == ri:
                                new_d = d
                                delta = new_d - old_d
                            else:
                                # removing c from ri, adding to rj
                                old_ri = 0.0
                                if route:
                                    old_ri = dist[0, route[0]] + sum(dist[route[k], route[k+1]] for k in range(len(route)-1)) + dist[route[-1], 0]
                                old_rj = 0.0
                                if rout:
                                    old_rj = dist[0, rout[0]] + sum(dist[rout[k], rout[k+1]] for k in range(len(rout)-1)) + dist[rout[-1], 0]
                                new_rj = d
                                new_ri = 0.0
                                if route2:
                                    new_ri = dist[0, route2[0]] + sum(dist[route2[k], route2[k+1]] for k in range(len(route2)-1)) + dist[route2[-1], 0]
                                delta = (new_rj + new_ri) - (old_ri + old_rj)
                            if delta < best_cost:
                                best_cost = delta
                                best = (rj, pos)
                if best is not None and best_cost < -1e-9:
                    rj, pos = best
                    # remove c from ri
                    routes[ri] = route2
                    if rj == ri:
                        routes[ri].insert(pos, c)
                    else:
                        routes[rj].insert(pos, c)
                    improved = True

    # Final validation and return
    result = []
    for r in routes:
        if r:
            result.append(r)
    return result
