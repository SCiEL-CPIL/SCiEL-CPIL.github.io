"""EV routing with time windows: construct + battery-aware local search."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_DEADLINE = 150.0
_MARGIN = 5.0


def _euclid(coords):
    n = len(coords)
    d = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            dd = math.hypot(dx, dy)
            d[i, j] = dd
            d[j, i] = dd
    return d


def _check_route(route, D, inst):
    n = len(inst["coords"])
    ready = inst["ready"]
    due = inst["due"]
    service = inst["service"]
    demands = inst["demands"]
    stations = set(inst["stations"])
    cap = inst["capacity"]
    battery = inst["battery"]
    speed = inst["speed"]
    rate = inst["recharge_rate"]
    horizon = inst["horizon"]

    t = 0.0
    bat = battery
    load = 0
    prev = 0
    for v in route:
        dist = D[prev, v]
        bat -= dist
        t += dist / speed
        if bat < -1e-9:
            return False
        if v in stations:
            need = max(0.0, battery - bat)
            t += need * rate
            bat = battery
            prev = v
            continue
        if load + demands[v] > cap + 1e-9:
            return False
        if t < ready[v]:
            t = ready[v]
        if t > due[v] + 1e-9:
            return False
        t += service[v]
        load += demands[v]
        prev = v
    dist = D[prev, 0]
    bat -= dist
    t += dist / speed
    if bat < -1e-9 or t > horizon + 1e-9:
        return False
    return True


def _route_cost(route, D):
    if not route:
        return 0.0
    c = D[0, route[0]]
    for i in range(1, len(route)):
        c += D[route[i - 1], route[i]]
    c += D[route[-1], 0]
    return c


def _nearest_nearest_construct(customers, D, inst):
    ready = inst["ready"]
    due = inst["due"]
    service = inst["service"]
    demands = inst["demands"]
    stations = set(inst["stations"])
    cap = inst["capacity"]
    battery = inst["battery"]
    speed = inst["speed"]
    rate = inst["recharge_rate"]
    horizon = inst["horizon"]
    n = len(inst["coords"])

    remaining = list(customers)
    remaining.sort(key=lambda v: due[v])
    routes = []

    while remaining:
        # pick the customer with the earliest due time as seed
        seed = remaining.pop(0)
        route = [seed]
        load = demands[seed]
        # state after serving seed
        t = max(0.0, ready[seed]) + service[seed]
        bat = battery - D[0, seed]
        # battery after serving seed (no station)
        # greedy: while battery allows, add nearest feasible customer
        while True:
            # try to extend with nearest feasible customer
            best = None
            best_d = float('inf')
            last = route[-1]
            for c in remaining:
                d = D[last, c]
                # battery check: need to reach c
                if bat - d < -1e-9:
                    continue
                # capacity
                if load + demands[c] > cap + 1e-9:
                    continue
                # time check at c
                t_arr = t + d / speed
                if t_arr < ready[c]:
                    t_arr = ready[c]
                if t_arr > due[c] + 1e-9:
                    continue
                # after service, can we get back to depot with remaining battery?
                t_after = t_arr + service[c]
                bat_after = bat - d
                # check return trip
                d_back = D[c, 0]
                if bat_after - d_back < -1e-9:
                    continue
                if t_after + d_back / speed > horizon + 1e-9:
                    continue
                if d < best_d:
                    best_d = d
                    best = c
            if best is None:
                break
            remaining.remove(best)
            route.append(best)
            load += demands[best]
            t += best_d / speed
            if t < ready[best]:
                t = ready[best]
            t += service[best]
            bat -= best_d

        # ensure feasibility of the constructed route; if not, try to fix by splitting
        if not _check_route(route, D, inst):
            # fallback: serve each customer in route individually
            for v in route:
                routes.append([v])
        else:
            routes.append(route)

    return routes


def _try_relocate(routes, D, inst):
    """Try to move a customer from one route to another. Return improvement or 0."""
    best_imp = 0.0
    best_op = None
    for i in range(len(routes)):
        for pos in range(len(routes[i])):
            c = routes[i][pos]
            # try inserting into each other route
            for j in range(len(routes)):
                if i == j:
                    continue
                for k in range(len(routes[j]) + 1):
                    new_j = routes[j][:k] + [c] + routes[j][k:]
                    new_i = routes[i][:pos] + routes[i][pos + 1:]
                    # quick cost check
                    old_cost = _route_cost(routes[i], D) + _route_cost(routes[j], D)
                    new_cost = _route_cost(new_i, D) + _route_cost(new_j, D)
                    imp = old_cost - new_cost
                    if imp <= best_imp:
                        continue
                    if not _check_route(new_j, D, inst):
                        continue
                    if new_i and not _check_route(new_i, D, inst):
                        continue
                    best_imp = imp
                    best_op = (i, pos, j, k)
    if best_op is not None:
        i, pos, j, k = best_op
        c = routes[i].pop(pos)
        routes[j].insert(k, c)
        return best_imp
    return 0.0


def _try_2opt(route, D, inst):
    """2-opt within a route. Return improvement."""
    best_imp = 0.0
    best_seg = None
    n = len(route)
    for i in range(n - 1):
        for j in range(i + 1, n):
            # reverse segment i..j
            new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
            old_cost = _route_cost(route, D)
            new_cost = _route_cost(new_route, D)
            imp = old_cost - new_cost
            if imp <= best_imp:
                continue
            if not _check_route(new_route, D, inst):
                continue
            best_imp = imp
            best_seg = (i, j)
    if best_seg is not None:
        i, j = best_seg
        route[i:j + 1] = route[i:j + 1][::-1]
        return best_imp
    return 0.0


def _try_swap(routes, D, inst):
    """Swap two customers between routes. Return improvement."""
    best_imp = 0.0
    best_op = None
    for i in range(len(routes)):
        for j in range(i + 1, len(routes)):
            for pi in range(len(routes[i])):
                for pj in range(len(routes[j])):
                    a = routes[i][pi]
                    b = routes[j][pj]
                    new_i = list(routes[i])
                    new_j = list(routes[j])
                    new_i[pi] = b
                    new_j[pj] = a
                    old_cost = _route_cost(routes[i], D) + _route_cost(routes[j], D)
                    new_cost = _route_cost(new_i, D) + _route_cost(new_j, D)
                    imp = old_cost - new_cost
                    if imp <= best_imp:
                        continue
                    if not _check_route(new_i, D, inst):
                        continue
                    if not _check_route(new_j, D, inst):
                        continue
                    best_imp = imp
                    best_op = (i, pi, j, pj)
    if best_op is not None:
        i, pi, j, pj = best_op
        routes[i][pi], routes[j][pj] = routes[j][pj], routes[i][pi]
        return best_imp
    return 0.0


def plan(instance):
    t0 = time.time()
    coords = instance["coords"]
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    D = _euclid(coords)

    # Build initial solution
    routes = _nearest_nearest_construct(customers, D, instance)

    # Verify all customers served exactly once
    served = []
    for r in routes:
        served.extend(r)
    if set(served) != set(customers) or len(served) != len(customers):
        # fallback: one per route
        routes = [[v] for v in customers]

    # Local search
    deadline = t0 + _DEADLINE - _MARGIN
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # 2-opt within each route
        for i in range(len(routes)):
            if time.time() > deadline:
                break
            imp = _try_2opt(routes[i], D, instance)
            if imp > 1e-9:
                improved = True
        if time.time() > deadline:
            break
        # relocate
        imp = _try_relocate(routes, D, instance)
        if imp > 1e-9:
            improved = True
        if time.time() > deadline:
            break
        # swap
        imp = _try_swap(routes, D, instance)
        if imp > 1e-9:
            improved = True

    # Final feasibility check; if broken, fall back
    for r in routes:
        if not _check_route(r, D, instance):
            routes = [[v] for v in customers]
            break

    return routes
# EVOLVE-BLOCK-END
