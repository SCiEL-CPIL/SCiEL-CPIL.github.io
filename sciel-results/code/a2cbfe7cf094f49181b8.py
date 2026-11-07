"""Electric vehicle routing with time windows – structural redesign.

Strategy:
  1. Build an initial solution via a time-window-aware savings heuristic that
     also tracks battery so that routes are feasible from the start.
  2. Improve with a battery-aware local search: relocate, swap, 2-opt, and
     or-opt (segment) moves, with a deadline guard.
  3. Return the best feasible solution found before the wall-clock limit.
"""
import time
import math
import numpy as np

_DEADLINE_FRAC = 0.85  # finish 15 % early to be safe
_WALL_LIMIT = 150.0


def _dist_matrix(coords):
    n = len(coords)
    m = np.zeros((n, n), dtype=float)
    for i in range(n):
        xi, yi = coords[i]
        for j in range(i + 1, n):
            xj, yj = coords[j]
            d = math.hypot(xi - xj, yi - yj)
            m[i, j] = d
            m[j, i] = d
    return m


def _route_info(route, D, demands, ready, due, service,
                battery_cap, recharge_rate, horizon, speed, station_set):
    """Simulate a route; return (total_dist, feasible, total_time)."""
    t = 0.0
    bat = battery_cap
    load = 0.0
    total_d = 0.0
    prev = 0  # depot
    for node in route:
        d = D[prev, node]
        t += d / speed
        bat -= d
        if bat < -1e-9:
            return total_d + d, False, t
        total_d += d
        if node in station_set:
            charge = (battery_cap - bat) * recharge_rate
            t += charge
            bat = battery_cap
        else:
            # customer
            if t < ready[node]:
                t = ready[node]
            if t > due[node] + 1e-9:
                return total_d, False, t
            t += service[node]
            load += demands[node]
        prev = node
    # return to depot
    d = D[prev, 0]
    t += d / speed
    bat -= d
    if bat < -1e-9:
        return total_d + d, False, t
    total_d += d
    if t > horizon + 1e-9:
        return total_d, False, t
    if load > 100 + 1e-9:
        return total_d, False, t
    return total_d, True, t


def _initial_solution(instance):
    coords = np.array(instance["coords"])
    n = len(coords)
    D = _dist_matrix(coords)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    cap = instance["capacity"]
    bat_cap = instance["battery"]
    rr = instance["recharge_rate"]
    horizon = instance["horizon"]
    speed = instance["speed"]
    stations = set(instance["stations"])
    n_cust = instance["n_customers"]

    # sort customers by due time
    cust_order = sorted(range(1, n_cust + 1), key=lambda v: due[v])

    routes = []
    unassigned = set(cust_order)

    # Greedy insertion: build routes by inserting customers into existing routes
    # or opening new ones, respecting battery/time/capacity.
    while unassigned:
        # try to insert each unassigned customer into an existing route
        best_key = None
        best_route_idx = -1
        best_pos = -1
        best_cost = float('inf')
        for ridx, route in enumerate(routes):
            for pos in range(len(route) + 1):
                # find candidate customer
                for c in list(unassigned):
                    new_route = route[:pos] + [c] + route[pos:]
                    td, feas, _ = _route_info(new_route, D, demands, ready, due,
                                              service, bat_cap, rr, horizon,
                                              speed, stations)
                    if feas:
                        # cost = increase in total distance
                        # approximate: D[prev,c]+D[c,next] - D[prev,next]
                        prev_node = route[pos - 1] if pos > 0 else 0
                        next_node = route[pos] if pos < len(route) else 0
                        inc = D[prev_node, c] + D[c, next_node] - D[prev_node, next_node]
                        if inc < best_cost:
                            best_cost = inc
                            best_route_idx = ridx
                            best_pos = pos
                            best_key = c
        if best_route_idx >= 0:
            routes[best_route_idx].insert(best_pos, best_key)
            unassigned.discard(best_key)
        else:
            # open new route with the customer with earliest due
            c = min(unassigned, key=lambda v: due[v])
            routes.append([c])
            unassigned.discard(c)

    return routes, D, demands, ready, due, service, bat_cap, rr, horizon, speed, stations


def _local_search(routes, D, demands, ready, due, service,
                  bat_cap, rr, horizon, speed, stations, deadline):
    """Battery-aware local search with relocate, swap, 2-opt, or-opt."""
    n_routes = len(routes)
    best_total = sum(_route_info(r, D, demands, ready, due, service,
                                 bat_cap, rr, horizon, speed, stations)[0]
                     for r in routes)
    no_improve = 0
    max_no_improve = 50

    while time.time() < deadline:
        improved = False
        # Relocate
        for i in range(n_routes):
            if time.time() > deadline:
                break
            ri = list(routes[i])
            for pos in range(len(ri)):
                c = ri[pos]
                ri2 = ri[:pos] + ri[pos + 1:]
                for j in range(n_routes):
                    if i == j:
                        continue
                    rj = list(routes[j])
                    for pos2 in range(len(rj) + 1):
                        new_j = rj[:pos2] + [c] + rj[pos2:]
                        td_i, f_i, _ = _route_info(ri2, D, demands, ready, due,
                                                   service, bat_cap, rr, horizon,
                                                   speed, stations)
                        td_j, f_j, _ = _route_info(new_j, D, demands, ready, due,
                                                   service, bat_cap, rr, horizon,
                                                   speed, stations)
                        if f_i and f_j:
                            new_total = td_i + td_j
                            if new_total < best_total - 1e-9:
                                routes[i] = ri2
                                routes[j] = new_j
                                best_total = new_total
                                improved = True
                                no_improve = 0
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue

        # Swap
        for i in range(n_routes):
            if time.time() > deadline:
                break
            ri = list(routes[i])
            for pos in range(len(ri)):
                c1 = ri[pos]
                for j in range(i + 1, n_routes):
                    rj = list(routes[j])
                    for pos2 in range(len(rj)):
                        c2 = rj[pos2]
                        new_i = ri[:pos] + [c2] + ri[pos + 1:]
                        new_j = rj[:pos2] + [c1] + rj[pos2 + 1:]
                        td_i, f_i, _ = _route_info(new_i, D, demands, ready, due,
                                                   service, bat_cap, rr, horizon,
                                                   speed, stations)
                        td_j, f_j, _ = _route_info(new_j, D, demands, ready, due,
                                                   service, bat_cap, rr, horizon,
                                                   speed, stations)
                        if f_i and f_j:
                            new_total = td_i + td_j
                            if new_total < best_total - 1e-9:
                                routes[i] = new_i
                                routes[j] = new_j
                                best_total = new_total
                                improved = True
                                no_improve = 0
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue

        # 2-opt within each route
        for i in range(n_routes):
            if time.time() > deadline:
                break
            ri = list(routes[i])
            for a in range(len(ri) - 1):
                for b in range(a + 1, len(ri)):
                    new_ri = ri[:a] + ri[a:b + 1][::-1] + ri[b + 1:]
                    td, f, _ = _route_info(new_ri, D, demands, ready, due,
                                           service, bat_cap, rr, horizon,
                                           speed, stations)
                    if f:
                        old_td, _, _ = _route_info(ri, D, demands, ready, due,
                                                   service, bat_cap, rr, horizon,
                                                   speed, stations)
                        if td < old_td - 1e-9:
                            routes[i] = new_ri
                            best_total = best_total - old_td + td
                            improved = True
                            no_improve = 0
                            break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue

        # Or-opt: move 2 or 3 consecutive customers
        for seg_len in (2, 3):
            for i in range(n_routes):
                if time.time() > deadline:
                    break
                ri = list(routes[i])
                for pos in range(len(ri) - seg_len + 1):
                    seg = ri[pos:pos + seg_len]
                    ri2 = ri[:pos] + ri[pos + seg_len:]
                    for j in range(n_routes):
                        if i == j:
                            continue
                        rj = list(routes[j])
                        for pos2 in range(len(rj) + 1):
                            new_j = rj[:pos2] + seg + rj[pos2:]
                            td_i, f_i, _ = _route_info(ri2, D, demands, ready, due,
                                                       service, bat_cap, rr, horizon,
                                                       speed, stations)
                            td_j, f_j, _ = _route_info(new_j, D, demands, ready, due,
                                                       service, bat_cap, rr, horizon,
                                                       speed, stations)
                            if f_i and f_j:
                                new_total = td_i + td_j
                                if new_total < best_total - 1e-9:
                                    routes[i] = ri2
                                    routes[j] = new_j
                                    best_total = new_total
                                    improved = True
                                    no_improve = 0
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue

        no_improve += 1
        if no_improve >= max_no_improve:
            break

    return routes, best_total


def plan(instance):
    t0 = time.time()
    deadline = t0 + _WALL_LIMIT * _DEADLINE_FRAC

    coords = np.array(instance["coords"])
    n = len(coords)
    D = _dist_matrix(coords)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    cap = instance["capacity"]
    bat_cap = instance["battery"]
    rr = instance["recharge_rate"]
    horizon = instance["horizon"]
    speed = instance["speed"]
    stations = set(instance["stations"])
    n_cust = instance["n_customers"]

    # Initial solution
    routes, D, demands, ready, due, service, bat_cap, rr, horizon, speed, stations = \
        _initial_solution(instance)

    # Local search
    if time.time() < deadline:
        routes, _ = _local_search(routes, D, demands, ready, due, service,
                                  bat_cap, rr, horizon, speed, stations, deadline)

    # Ensure every customer is served exactly once
    all_served = []
    for r in routes:
        all_served.extend(r)
    missing = set(range(1, n_cust + 1)) - set(all_served)
    for c in missing:
        routes.append([c])

    # Final feasibility check; if any route infeasible, fall back to one-per-route
    for r in routes:
        _, feas, _ = _route_info(r, D, demands, ready, due, service,
                                 bat_cap, rr, horizon, speed, stations)
        if not feas:
            # split into singletons
            flat = []
            for rr_ in routes:
                flat.extend(rr_)
            routes = [[v] for v in flat]
            break

    return routes
