"""EV routing with time windows: constructive + relocate/2-opt/Or-opt local search."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _eval_route(seq, coords, due, ready, service, stations, battery,
                recharge_rate, horizon, speed):
    """Return (ok, total_distance, final_time, battery_at_depot)."""
    c = coords
    t = 0.0
    b = float(battery)
    total = 0.0
    prev = 0
    for node in seq:
        d = _dist(c[prev], c[node])
        t += d / speed
        b -= d
        total += d
        if b < -1e-9:
            return False, total, t, b
        if node in stations:
            if b < 0.0:
                b = 0.0
            t += (-b) * recharge_rate
            b = float(battery)
        else:
            t = max(t, ready[node])
            if t > due[node] + 1e-9:
                return False, total, t, b
            t += service[node]
        prev = node
    d = _dist(c[prev], c[0])
    t += d / speed
    b -= d
    total += d
    if b < -1e-9 or t > horizon + 1e-9:
        return False, total, t, b
    return True, total, t, b


def _eval_routes(routes, coords, due, ready, service, stations, battery,
                 recharge_rate, horizon, speed):
    total = 0.0
    for r in routes:
        ok, d, _, _ = _eval_route(r, coords, due, ready, service, stations,
                                  battery, recharge_rate, horizon, speed)
        if not ok:
            return None
        total += d
    return total


def _construct(coords, n, demands, ready, due, service, stations, battery,
               recharge_rate, horizon, speed, capacity):
    """Cluster by capacity, then nearest-neighbour with feasibility checks."""
    cust = list(range(1, n + 1))
    cust.sort(key=lambda v: due[v])
    clusters = []
    cur = []
    cur_cap = 0
    for v in cust:
        if cur and cur_cap + demands[v] > capacity:
            clusters.append(cur)
            cur = []
            cur_cap = 0
        cur.append(v)
        cur_cap += demands[v]
    if cur:
        clusters.append(cur)

    routes = []
    for cl in clusters:
        rem = list(cl)
        route = []
        prev = 0
        t = 0.0
        b = float(battery)
        while rem:
            best = -1
            best_cost = None
            for v in rem:
                ok, d, t2, b2 = _eval_route(
                    route + [v], coords, due, ready, service, stations,
                    battery, recharge_rate, horizon, speed)
                if ok:
                    cost = d + _dist(coords[v], coords[0])
                    if best_cost is None or cost < best_cost:
                        best_cost = cost
                        best = v
            if best < 0:
                # cannot extend feasibly; start a new route with the earliest-due
                if route:
                    routes.append(route)
                    route = []
                    prev = 0
                    t = 0.0
                    b = float(battery)
                # force the earliest-due remaining customer alone
                rem.sort(key=lambda v: due[v])
                best = rem[0]
            rem.remove(best)
            route.append(best)
        if route:
            routes.append(route)
    return routes


def _routes_valid(routes, coords, due, ready, service, stations, battery,
                  recharge_rate, horizon, speed, capacity):
    served = [False] * (len(coords))
    for r in routes:
        cap = 0
        for node in r:
            if served[node]:
                return False
            served[node] = True
            cap += service[node] * 0 + 0  # placeholder
        # capacity by demand handled separately
    for i in range(1, len(coords)):
        if not served[i]:
            return False
    return True


def _solve_instance(instance, deadline):
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    n = instance["n_customers"]

    # precompute demand-based capacity check helper
    def cap_ok(route):
        return sum(demands[v] for v in route) <= capacity

    routes = _construct(coords, n, demands, ready, due, service, stations,
                        battery, recharge_rate, horizon, speed, capacity)
    # ensure every customer served exactly once (fallback: singleton routes)
    served = [False] * len(coords)
    for r in routes:
        for v in r:
            served[v] = True
    for v in range(1, n + 1):
        if not served[v]:
            routes.append([v])

    def route_cost(r):
        ok, d, _, _ = _eval_route(r, coords, due, ready, service, stations,
                                  battery, recharge_rate, horizon, speed)
        return d if ok else float('inf')

    def total_cost(routes):
        return sum(route_cost(r) for r in routes)

    best = routes
    best_c = total_cost(routes)

    # local search: relocate + 2-opt + Or-opt(1-3), with validity gates
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for ri in range(len(routes)):
            r = routes[ri]
            # 2-opt within route
            for i in range(len(r) - 1):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(r)):
                    newr = r[:i] + r[i:j + 1][::-1] + r[j + 1:]
                    if not cap_ok(newr):
                        continue
                    c = route_cost(newr)
                    if c < route_cost(r) - 1e-9:
                        routes[ri] = newr
                        r = newr
                        improved = True
            r = routes[ri]
            # Or-opt (segment length 1-3) within route
            for L in (1, 2, 3):
                for i in range(len(r) - L + 1):
                    if time.time() > deadline:
                        break
                    seg = r[i:i + L]
                    rest = r[:i] + r[i + L:]
                    for j in range(len(rest) + 1):
                        newr = rest[:j] + seg + rest[j:]
                        if newr == r:
                            continue
                        if not cap_ok(newr):
                            continue
                        c = route_cost(newr)
                        if c < route_cost(r) - 1e-9:
                            routes[ri] = newr
                            r = newr
                            improved = True
            r = routes[ri]
            # relocate between routes
            for v in list(r):
                if time.time() > deadline:
                    break
                rest = [x for x in r if x != v]
                for rj in range(len(routes)):
                    if rj == ri:
                        continue
                    target = routes[rj]
                    for pos in range(len(target) + 1):
                        new_target = target[:pos] + [v] + target[pos:]
                        if not cap_ok(new_target):
                            continue
                        if not cap_ok(rest):
                            continue
                        c_old = route_cost(r) + route_cost(target)
                        c_new = route_cost(rest) + route_cost(new_target)
                        if c_new < c_old - 1e-9:
                            routes[ri] = rest
                            routes[rj] = new_target
                            improved = True
                            break
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break

    # final validity gate; if anything broke, fall back to safe singletons
    c = total_cost(best if best_c <= total_cost(routes) else routes)
    final = best if best_c <= total_cost(routes) else routes
    ok, tot, _, _ = _eval_routes(final, coords, due, ready, service, stations,
                                 battery, recharge_rate, horizon, speed)
    if not ok:
        final = [[v] for v in range(1, n + 1)]
    return final


def plan(instance):
    start = time.time()
    deadline = start + 115.0  # safety margin under 120s
    result = _solve_instance(instance, deadline)
    return result
# EVOLVE-BLOCK-END
