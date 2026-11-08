"""EV routing with time windows, battery and charging."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 110.0  # safety margin below the 120s wall-clock limit


def _build_dist(coords):
    c = np.asarray(coords, dtype=float)
    d = np.sqrt(np.sum((c[:, None, :] - c[None, :, :]) ** 2, axis=2))
    return d


def _simulate(route, dist, ready, due, service, is_station, battery, recharge_rate, horizon):
    """Return (ok, total_dist)."""
    t = 0.0
    bat = float(battery)
    total = 0.0
    prev = 0
    for node in route:
        d = float(dist[prev, node])
        total += d
        t += d
        bat -= d
        if bat < -1e-7:
            return False, total
        if not is_station[node]:
            if t < ready[node]:
                t = float(ready[node])
            if t > due[node] + 1e-7:
                return False, total
            t += float(service[node])
        else:
            charge = (bat) * recharge_rate
            t += charge
            bat = float(battery)
        prev = node
    d = float(dist[prev, 0])
    total += d
    t += d
    bat -= d
    if bat < -1e-7 or t > horizon + 1e-7:
        return False, total
    return True, total


def _route_load(route, demands):
    return sum(float(demands[n]) for n in route)


def _solve_instance(inst, deadline):
    coords = inst["coords"]
    demands = inst["demands"]
    ready = inst["ready"]
    due = inst["due"]
    service = inst["service"]
    n_cust = inst["n_customers"]
    stations = set(inst["stations"])
    capacity = inst["capacity"]
    battery = inst["battery"]
    recharge_rate = inst["recharge_rate"]
    horizon = inst["horizon"]

    dist = _build_dist(coords)
    is_station = np.zeros(len(coords), dtype=bool)
    for s in stations:
        is_station[s] = True
    station_list = list(stations)

    # initial order: earliest due first
    customers = list(range(1, n_cust + 1))
    customers.sort(key=lambda v: due[v])

    # Greedy nearest-feasible insertion into routes
    routes = []
    for v in customers:
        placed = False
        best_cost = None
        best_route = None
        best_idx = None
        for ri, route in enumerate(routes):
            if _route_load(route, demands) + demands[v] > capacity + 1e-9:
                continue
            for idx in range(len(route) + 1):
                cand = route[:idx] + [v] + route[idx:]
                ok, c = _simulate(cand, dist, ready, due, service, is_station,
                                  battery, recharge_rate, horizon)
                if ok:
                    if best_cost is None or c < best_cost:
                        best_cost = c
                        best_route = ri
                        best_idx = idx
        if best_route is not None:
            routes[best_route] = routes[best_route][:best_idx] + [v] + routes[best_route][best_idx:]
            placed = True
        if not placed:
            routes.append([v])

    # Local search: 2-opt, relocate, swap
    def total_cost(routes):
        tot = 0.0
        for r in routes:
            _, c = _simulate(r, dist, ready, due, service, is_station,
                             battery, recharge_rate, horizon)
            tot += c
        return tot

    def try_2opt(route):
        n = len(route)
        improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 1, n):
                    seg = route[i:j + 1][::-1]
                    cand = route[:i] + seg + route[j + 1:]
                    ok, c = _simulate(cand, dist, ready, due, service, is_station,
                                      battery, recharge_rate, horizon)
                    if ok:
                        _, oc = _simulate(route, dist, ready, due, service, is_station,
                                          battery, recharge_rate, horizon)
                        if c < oc - 1e-9:
                            route[:] = cand
                            improved = True
                            break
                if improved:
                    break
        return route

    def try_relocate(routes):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for ri, route in enumerate(routes):
                for k in range(len(route)):
                    if time.time() > deadline:
                        break
                    node = route[k]
                    if is_station[node]:
                        continue
                    base = route[:k] + route[k + 1:]
                    for rj, other in enumerate(routes):
                        if rj == ri:
                            continue
                        if _route_load(other, demands) + demands[node] > capacity + 1e-9:
                            continue
                        for idx in range(len(other) + 1):
                            cand_other = other[:idx] + [node] + other[idx:]
                            ok, c = _simulate(cand_other, dist, ready, due, service,
                                              is_station, battery, recharge_rate, horizon)
                            if not ok:
                                continue
                            okb, cb = _simulate(base, dist, ready, due, service,
                                                is_station, battery, recharge_rate, horizon)
                            if not okb:
                                continue
                            _, c_old_o = _simulate(other, dist, ready, due, service,
                                                   is_station, battery, recharge_rate, horizon)
                            _, c_old_r = _simulate(route, dist, ready, due, service,
                                                   is_station, battery, recharge_rate, horizon)
                            if (c + cb) < (c_old_o + c_old_r) - 1e-9:
                                routes[ri] = base
                                routes[rj] = cand_other
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return routes

    def try_swap(routes):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for ri in range(len(routes)):
                for rj in range(ri + 1, len(routes)):
                    if time.time() > deadline:
                        break
                    a = routes[ri]
                    b = routes[rj]
                    for i in range(len(a)):
                        if is_station[a[i]]:
                            continue
                        for j in range(len(b)):
                            if is_station[b[j]]:
                                continue
                            if a[i] == b[j]:
                                continue
                            if _route_load(a, demands) - demands[a[i]] + demands[b[j]] > capacity + 1e-9:
                                continue
                            if _route_load(b, demands) - demands[b[j]] + demands[a[i]] > capacity + 1e-9:
                                continue
                            na = a[:i] + [b[j]] + a[i + 1:]
                            nb = b[:j] + [a[i]] + b[j + 1:]
                            oka, ca = _simulate(na, dist, ready, due, service, is_station,
                                                battery, recharge_rate, horizon)
                            okb, cb = _simulate(nb, dist, ready, due, service, is_station,
                                                battery, recharge_rate, horizon)
                            if oka and okb:
                                _, oa = _simulate(a, dist, ready, due, service, is_station,
                                                  battery, recharge_rate, horizon)
                                _, ob = _simulate(b, dist, ready, due, service, is_station,
                                                  battery, recharge_rate, horizon)
                                if (ca + cb) < (oa + ob) - 1e-9:
                                    routes[ri] = na
                                    routes[rj] = nb
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return routes

    # iterate improvement passes
    best_routes = None
    best_total = None
    while time.time() < deadline:
        for r in routes:
            try_2opt(r)
        routes = try_relocate(routes)
        for r in routes:
            try_2opt(r)
        routes = try_swap(routes)
        for r in routes:
            try_2opt(r)
        tot = total_cost(routes)
        if best_total is None or tot < best_total - 1e-9:
            best_total = tot
            best_routes = [list(r) for r in routes]
        if time.time() > deadline:
            break

    if best_routes is None:
        best_routes = routes
    return best_routes


def plan(instance):
    t0 = time.time()
    deadline = t0 + _DEADLINE
    return _solve_instance(instance, deadline)
# EVOLVE-BLOCK-END
