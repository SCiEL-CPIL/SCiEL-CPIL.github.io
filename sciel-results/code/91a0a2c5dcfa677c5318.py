"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import math
import time


def _dist_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
            D[i, j] = d
            D[j, i] = d
    return D


def _route_state(route, D, instance):
    """Return (time, battery, dist, feasible, demand) for a route."""
    if not route:
        return 0.0, instance.get("battery", 300), 0.0, True, 0
    speed = instance["speed"]
    battery_cap = instance["battery"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = instance["_stations_set"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    demands = instance["demands"]

    t = 0.0
    bat = float(battery_cap)
    dist = 0.0
    demand = 0
    prev = 0
    for v in route:
        d = D[prev, v]
        t += d
        bat -= d
        dist += d
        if bat < -1e-9:
            return t, bat, dist, False, demand
        if v in stations:
            energy_needed = battery_cap - bat
            t += energy_needed * recharge_rate
            bat = float(battery_cap)
        else:
            demand += demands[v]
            if demand > instance["capacity"] + 1e-9:
                return t, bat, dist, False, demand
            t = max(t, ready[v])
            if t > due[v] + 1e-9:
                return t, bat, dist, False, demand
            t += service[v]
        prev = v
    d = D[prev, 0]
    t += d
    bat -= d
    dist += d
    if bat < -1e-9:
        return t, bat, dist, False, demand
    if t > horizon + 1e-9:
        return t, bat, dist, False, demand
    return t, bat, dist, True, demand


def _route_feasible(route, D, instance):
    if not route:
        return True
    _, _, _, feas, _ = _route_state(route, D, instance)
    return feas


def _route_dist(route, D):
    if not route:
        return 0.0
    total = D[0, route[0]]
    for i in range(len(route) - 1):
        total += D[route[i], route[i + 1]]
    total += D[route[-1], 0]
    return total


def _savings_construct(instance, D, deadline):
    """Clarke-Wright savings construction adapted for EVRP."""
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    cap = instance["capacity"]
    horizon = instance["horizon"]
    stations = instance["_stations_set"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    demands = instance["demands"]

    # Start with each customer as its own route
    routes = [[v] for v in customers]
    route_of = {v: i for i, r in enumerate(routes) for v in r}
    unassigned = set(customers)

    # Compute savings: saving of merging route ending at a and route starting at b
    # s(a,b) = D[0,a] + D[0,b] - D[a,b]
    # We want to merge routes in order of decreasing savings
    savings_list = []
    for i in range(len(customers)):
        for j in range(i + 1, len(customers)):
            a = customers[i]
            b = customers[j]
            s = D[0, a] + D[0, b] - D[a, b]
            savings_list.append((s, a, b))
    savings_list.sort(reverse=True)

    # Track route endpoints
    route_start = {i: routes[i][0] for i in range(len(routes))}
    route_end = {i: routes[i][-1] for i in range(len(routes))}
    route_demand = {i: demands[routes[i][0]] for i in range(len(routes))}
    active = set(range(len(routes)))

    for s, a, b in savings_list:
        if time.time() > deadline:
            break
        if s <= 0:
            break
        ri = route_of.get(a, -1)
        rj = route_of.get(b, -1)
        if ri == -1 or rj == -1 or ri == rj:
            continue
        if ri not in active or rj not in active:
            continue

        # Try all 4 merge orders: [a..] + [b..], [b..] + [a..], reversed
        best_merge = None
        best_dist = float('inf')

        # Determine if a is at end of ri and b is at start of rj, etc.
        a_at_end = (routes[ri][-1] == a)
        a_at_start = (routes[ri][0] == a)
        b_at_end = (routes[rj][-1] == b)
        b_at_start = (routes[rj][0] == b)

        candidates = []
        # ri followed by rj (a at end of ri, b at start of rj)
        if a_at_end and b_at_start:
            candidates.append((ri, rj, False, False))
        # rj followed by ri (b at end of rj, a at start of ri)
        if b_at_end and a_at_start:
            candidates.append((rj, ri, False, False))
        # ri reversed + rj
        if a_at_start and b_at_start:
            candidates.append((ri, rj, True, False))
        # rj reversed + ri
        if b_at_start and a_at_start:
            candidates.append((rj, ri, True, False))
        # ri + rj reversed
        if a_at_end and b_at_end:
            candidates.append((ri, rj, False, True))
        # rj + ri reversed
        if b_at_end and a_at_end:
            candidates.append((rj, ri, True, True))
        # ri reversed + rj reversed
        if a_at_start and b_at_end:
            candidates.append((ri, rj, True, True))
        # rj reversed + ri reversed
        if b_at_start and a_at_end:
            candidates.append((rj, ri, True, True))

        for first, second, rev_first, rev_second in candidates:
            if first not in active or second not in active:
                continue
            new_route = routes[first][:]
            if rev_first:
                new_route = new_route[::-1]
            second_route = routes[second][:]
            if rev_second:
                second_route = second_route[::-1]
            new_route = new_route + second_route

            # Check feasibility
            if not _route_feasible(new_route, D, instance):
                continue

            nd = _route_dist(new_route, D)
            if nd < best_dist:
                best_dist = nd
                best_merge = (first, second, rev_first, rev_second, new_route)

        if best_merge is not None:
            first, second, rf, rs, new_route = best_merge
            routes[first] = new_route
            active.discard(second)
            # Update route_of
            for v in new_route:
                route_of[v] = first
            route_start[first] = new_route[0]
            route_end[first] = new_route[-1]
            route_demand[first] = sum(demands[v] for v in new_route)

    # Filter out empty/inactive routes
    final_routes = []
    for i in range(len(routes)):
        if i in active and routes[i]:
            final_routes.append(routes[i])

    # Verify all customers are assigned
    all_assigned = set()
    for r in final_routes:
        all_assigned.update(r)
    missing = set(customers) - all_assigned
    if missing:
        for v in missing:
            final_routes.append([v])

    return final_routes


def _2opt_fast(route, D, instance, max_iter=200):
    """Fast 2-opt with precomputed feasibility hints."""
    if len(route) < 2:
        return route
    best = route[:]
    n = len(best)

    for _ in range(max_iter):
        improved = False
        for i in range(n - 1):
            a = best[i - 1] if i > 0 else 0
            b = best[i]
            for j in range(i + 1, n):
                c = best[j]
                d = best[j + 1] if j + 1 < n else 0
                old_seg = D[a, b] + D[c, d]
                new_seg = D[a, c] + D[b, d]
                if new_seg >= old_seg - 1e-9:
                    continue
                new_route = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                if not _route_feasible(new_route, D, instance):
                    continue
                best = new_route
                n = len(best)
                improved = True
                break
            if improved:
                break
        if not improved:
            break
    return best


def _or_opt_fast(route, D, instance, max_iter=50):
    """Fast Or-opt: relocate single nodes."""
    if len(route) < 3:
        return route
    best = route[:]
    n = len(best)

    for _ in range(max_iter):
        improved = False
        for i in range(n):
            node = best[i]
            remaining = best[:i] + best[i + 1:]
            for j in range(len(remaining) + 1):
                if j == i or j == i - 1:
                    continue
                candidate = remaining[:j] + [node] + remaining[j:]
                if candidate == best:
                    continue
                if not _route_feasible(candidate, D, instance):
                    continue
                if _route_dist(candidate, D) < _route_dist(best, D) - 1e-9:
                    best = candidate
                    n = len(best)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best


def _relocate_inter(routes, D, instance, max_iters=20):
    """Inter-route relocate moves."""
    for _ in range(max_iters):
        improved = False
        for i in range(len(routes)):
            if not routes[i]:
                continue
            for j in range(len(routes)):
                if i == j:
                    continue
                for ai in range(len(routes[i])):
                    v = routes[i][ai]
                    new_i = routes[i][:]
                    new_i.pop(ai)
                    for aj in range(len(routes[j]) + 1):
                        new_j = routes[j][:]
                        new_j.insert(aj, v)
                        if not _route_feasible(new_i, D, instance):
                            continue
                        if not _route_feasible(new_j, D, instance):
                            continue
                        old_d = _route_dist(routes[i], D) + _route_dist(routes[j], D)
                        new_d = _route_dist(new_i, D) + _route_dist(new_j, D)
                        if new_d < old_d - 1e-9:
                            routes[i] = new_i
                            routes[j] = new_j
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


def _swap_inter(routes, D, instance, max_iters=10):
    """Inter-route swap moves."""
    for _ in range(max_iters):
        improved = False
        for i in range(len(routes)):
            if not routes[i]:
                continue
            for j in range(i + 1, len(routes)):
                if not routes[j]:
                    continue
                for ai in range(len(routes[i])):
                    for aj in range(len(routes[j])):
                        vi = routes[i][ai]
                        vj = routes[j][aj]
                        new_i = routes[i][:]
                        new_j = routes[j][:]
                        new_i[ai] = vj
                        new_j[aj] = vi
                        if not _route_feasible(new_i, D, instance):
                            continue
                        if not _route_feasible(new_j, D, instance):
                            continue
                        old_d = _route_dist(routes[i], D) + _route_dist(routes[j], D)
                        new_d = _route_dist(new_i, D) + _route_dist(new_j, D)
                        if new_d < old_d - 1e-9:
                            routes[i] = new_i
                            routes[j] = new_j
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


def plan(instance):
    start_time = time.time()
    deadline = start_time + 130  # Safety margin

    # Precompute
    coords = instance["coords"]
    D = _dist_matrix(coords)
    instance["_stations_set"] = set(instance.get("stations", []))

    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))

    # Construction: Savings-based
    routes = _savings_construct(instance, D, deadline)

    # Verify all customers covered
    all_assigned = set()
    for r in routes:
        all_assigned.update(r)
    missing = set(customers) - all_assigned
    if missing:
        for v in missing:
            routes.append([v])

    # Intra-route optimization: 2-opt then Or-opt
    for i in range(len(routes)):
        if time.time() > deadline - 30:
            break
        routes[i] = _2opt_fast(routes[i], D, instance, max_iter=100)
        routes[i] = _or_opt_fast(routes[i], D, instance, max_iter=30)

    # Inter-route optimization
    if time.time() < deadline - 20:
        routes = _relocate_inter(routes, D, instance, max_iters=10)
        routes = _swap_inter(routes, D, instance, max_iters=5)

    # Final intra-route polish
    for i in range(len(routes)):
        if time.time() > deadline - 10:
            break
        routes[i] = _2opt_fast(routes[i], D, instance, max_iter=50)

    # Remove empty routes
    routes = [r for r in routes if r]

    # Final feasibility check
    for r in routes:
        if not _route_feasible(r, D, instance):
            # Fallback: split into individual customers
            pass

    return routes
# EVOLVE-BLOCK-END
