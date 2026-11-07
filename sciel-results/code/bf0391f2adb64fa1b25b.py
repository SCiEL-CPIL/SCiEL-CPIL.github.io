"""EV routing with time windows: savings construction + 2-opt/Or-opt local search."""
# EVOLVE-BLOCK-START
import math
import time

_EPS = 1e-9


def _build(instance):
    coords = instance["coords"]
    n = len(coords)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = coords[i]
        row = D[i]
        for j in range(n):
            if i == j:
                continue
            xj, yj = coords[j]
            row[j] = math.hypot(xi - xj, yi - yj)
    return D


def _simulate(route, instance, D, stations_set):
    """Return (ok, total_distance)."""
    if not route:
        return True, 0.0
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge = instance["recharge_rate"]
    horizon = instance["horizon"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    demands = instance["demands"]

    load = 0.0
    battery = float(battery_max)
    t = 0.0
    prev = 0
    total = 0.0

    for node in route:
        d = D[prev][node]
        if d > battery + _EPS:
            return False, total
        battery -= d
        t += d / speed
        total += d
        prev = node
        if node in stations_set:
            if battery < 0.0:
                return False, total
            charge = battery_max - battery
            if charge > 0.0:
                t += charge * recharge
            battery = float(battery_max)
            continue
        load += demands[node]
        if load > capacity + _EPS:
            return False, total
        if t < ready[node]:
            t = float(ready[node])
        if t > due[node] + _EPS:
            return False, total
        t += service[node]
        if t > horizon + _EPS:
            return False, total

    d = D[prev][0]
    if d > battery + _EPS:
        return False, total
    battery -= d
    t += d / speed
    total += d
    if t > horizon + _EPS:
        return False, total
    return True, total


def _route_dist(route, D):
    if not route:
        return 0.0
    total = D[0][route[0]]
    for i in range(1, len(route)):
        total += D[route[i - 1]][route[i]]
    total += D[route[-1]][0]
    return total


def _try_add_station(route, instance, D, stations_set):
    """Try inserting a charging station to make route feasible. Returns fixed route or None."""
    n = len(route)
    for s in instance["stations"]:
        for p in range(n + 1):
            cand = route[:p] + [s] + route[p:]
            ok, _ = _simulate(cand, instance, D, stations_set)
            if ok:
                return cand
    return None


def _savings_construct(instance, D, stations_set, deadline):
    """Clarke-Wright savings heuristic with charging station insertion."""
    n_cust = instance["n_customers"]
    customers = list(range(1, n_cust + 1))
    capacity = instance["capacity"]
    demands = instance["demands"]

    # Start with each customer as its own route
    # Check individual feasibility
    routes = []
    for c in customers:
        single = [c]
        ok, _ = _simulate(single, instance, D, stations_set)
        if ok:
            routes.append(single)
        else:
            # Try with a station
            fixed = _try_add_station(single, instance, D, stations_set)
            if fixed is not None:
                routes.append(fixed)
            else:
                # Shouldn't happen with battery 300 and coords in [0,100]^2
                routes.append(single)

    # Compute savings for merging two routes
    # savings(i, j) = D[0][i] + D[0][j] - D[i][j]
    # We merge route ending at i with route starting at j
    # Each route is a list; we track endpoints
    # For savings, we need the "end" of one route and "start" of another
    # In a route [a, b, c], end is c, start is a
    # Merging: route1 + route2 means end of route1 connects to start of route2
    # But we can also reverse routes

    # Build savings list: for each pair of routes, compute best merge saving
    # Route i ends at routes[i][-1], starts at routes[i][0]
    # Merge saving = D[0][end_i] + D[0][start_j] - D[end_i][start_j]
    # Or reverse: D[0][end_i] + D[0][end_j] - D[end_i][end_j] (reverse route j)
    # Or: D[0][start_i] + D[0][start_j] - D[start_i][start_j] (reverse route i)
    # Or: D[0][start_i] + D[0][end_j] - D[start_i][end_j] (reverse both)

    # For simplicity, compute savings considering all 4 orientations
    # But standard CW: savings[i][j] = D[0][i] + D[0][j] - D[i][j]
    # where i is the end of route A, j is the start of route B

    # We'll use a priority approach: compute all possible merges, pick best feasible
    t0 = time.time()
    while time.time() < deadline:
        best_saving = 0.0
        best_pair = None  # (ri, rj, orientation)
        nr = len(routes)
        for ri in range(nr):
            for rj in range(ri + 1, nr):
                if time.time() - t0 > deadline - t0:
                    break
                ra = routes[ri]
                rb = routes[rj]
                # Check demand
                da = sum(demands[n] for n in ra if n not in stations_set)
                db = sum(demands[n] for n in rb if n not in stations_set)
                if da + db > capacity + _EPS:
                    continue
                # Try 4 orientations
                # 1: ra + rb
                # 2: ra + rb_reversed
                # 3: ra_reversed + rb
                # 4: ra_reversed + rb_reversed
                # Saving for merging = D[0][end_ra] + D[0][start_rb] - D[end_ra][start_rb]
                # For orientation 1: end_ra = ra[-1], start_rb = rb[0]
                # For orientation 2: end_ra = ra[-1], start_rb = rb[-1]
                # For orientation 3: end_ra = ra[0], start_rb = rb[0]
                # For orientation 4: end_ra = ra[0], start_rb = rb[-1]
                for orient in range(4):
                    if orient == 0:
                        end_a = ra[-1]
                        start_b = rb[0]
                        merged = ra + rb
                    elif orient == 1:
                        end_a = ra[-1]
                        start_b = rb[-1]
                        merged = ra + rb[::-1]
                    elif orient == 2:
                        end_a = ra[0]
                        start_b = rb[0]
                        merged = ra[::-1] + rb
                    else:
                        end_a = ra[0]
                        start_b = rb[-1]
                        merged = ra[::-1] + rb[::-1]

                    saving = D[0][end_a] + D[0][start_b] - D[end_a][start_b]
                    if saving <= best_saving + _EPS:
                        continue
                    # Check feasibility of merged route
                    ok, _ = _simulate(merged, instance, D, stations_set)
                    if ok:
                        best_saving = saving
                        best_pair = (ri, rj, orient)
                    else:
                        # Try with a station
                        fixed = _try_add_station(merged, instance, D, stations_set)
                        if fixed is not None:
                            # Check if the fixed route has better distance than the savings suggest
                            # Actually just check feasibility
                            best_saving = saving
                            best_pair = (ri, rj, orient, fixed)
        if best_pair is None:
            break
        ri, rj = best_pair[0], best_pair[1]
        orient = best_pair[2]
        ra = routes[ri]
        rb = routes[rj]
        if len(best_pair) > 3:
            merged = best_pair[3]
        else:
            if orient == 0:
                merged = ra + rb
            elif orient == 1:
                merged = ra + rb[::-1]
            elif orient == 2:
                merged = ra[::-1] + rb
            else:
                merged = ra[::-1] + rb[::-1]
        # Remove both routes, add merged
        # Remove the higher index first to preserve indices
        if ri > rj:
            ri, rj = rj, ri
        del routes[rj]
        del routes[ri]
        routes.append(merged)
    return routes


def _or_opt(routes, instance, D, stations_set, deadline):
    """Or-opt: move 1, 2, or 3 consecutive nodes within a route or to another route."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for ri in range(len(routes)):
            r = routes[ri]
            m = len(r)
            if m < 2:
                continue
            for seg_len in (1, 2, 3):
                if m < seg_len + 1:
                    continue
                for i in range(m - seg_len + 1):
                    if time.time() >= deadline:
                        return routes
                    seg = r[i:i + seg_len]
                    # Check if all are customers (not stations)
                    if any(n in stations_set for n in seg):
                        continue
                    rest = r[:i] + r[i + seg_len:]
                    # Try inserting segment at each position in rest
                    for j in range(len(rest) + 1):
                        if j == i or j == i + seg_len:
                            continue  # same position
                        cand = rest[:j] + seg + rest[j:]
                        ok, _ = _simulate(cand, instance, D, stations_set)
                        if ok:
                            new_d = _route_dist(cand, D)
                            old_d = _route_dist(r, D)
                            if new_d < old_d - _EPS:
                                routes[ri] = cand
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        # Cross-route Or-opt: move segment from route a to route b
        for a in range(len(routes)):
            ra = routes[a]
            m = len(ra)
            if m < 2:
                continue
            for seg_len in (1, 2):
                if m < seg_len + 1:
                    continue
                for i in range(m - seg_len + 1):
                    if time.time() >= deadline:
                        return routes
                    seg = ra[i:i + seg_len]
                    if any(n in stations_set for n in seg):
                        continue
                    rest_a = ra[:i] + ra[i + seg_len:]
                    if not rest_a:
                        continue
                    oka, _ = _simulate(rest_a, instance, D, stations_set)
                    if not oka:
                        continue
                    for b in range(len(routes)):
                        if b == a:
                            continue
                        rb = routes[b]
                        for j in range(len(rb) + 1):
                            cand_b = rb[:j] + seg + rb[j:]
                            okb, _ = _simulate(cand_b, instance, D, stations_set)
                            if okb:
                                new_total = _route_dist(cand_b, D) + _route_dist(rest_a, D)
                                old_total = _route_dist(ra, D) + _route_dist(rb, D)
                                if new_total < old_total - _EPS:
                                    routes[a] = rest_a
                                    routes[b] = cand_b
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
    return routes


def _two_opt(routes, instance, D, stations_set, deadline):
    """2-opt within each route."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for ri in range(len(routes)):
            r = routes[ri]
            m = len(r)
            if m < 3:
                continue
            for i in range(m - 1):
                if time.time() >= deadline:
                    return routes
                for j in range(i + 1, m):
                    cand = r[:i] + r[i:j + 1][::-1] + r[j + 1:]
                    ok, _ = _simulate(cand, instance, D, stations_set)
                    if ok and _route_dist(cand, D) < _route_dist(r, D) - _EPS:
                        routes[ri] = cand
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return routes


def _routes_valid(routes, instance, D, stations_set):
    seen = set()
    for r in routes:
        ok, _ = _simulate(r, instance, D, stations_set)
        if not ok:
            return False
        for node in r:
            if node in stations_set or node == 0:
                continue
            if node in seen:
                return False
            seen.add(node)
    n_cust = instance["n_customers"]
    if len(seen) != n_cust:
        return False
    return True


def plan(instance):
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill
    stations_set = set(instance["stations"])
    instance["stations_set"] = stations_set
    D = _build(instance)

    # Savings-based construction
    routes = _savings_construct(instance, D, stations_set, deadline)

    # Ensure all customers covered
    covered = set()
    for r in routes:
        for node in r:
            if node not in stations_set and node != 0:
                covered.add(node)
    missing = [v for v in range(1, instance["n_customers"] + 1) if v not in covered]
    for v in missing:
        routes.append([v])

    # Local search: 2-opt then Or-opt, alternating
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        routes = _two_opt(routes, instance, D, stations_set, time.time() + remaining_time * 0.4)
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        routes = _or_opt(routes, instance, D, stations_set, time.time() + remaining_time * 0.5)
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        routes = _two_opt(routes, instance, D, stations_set, time.time() + remaining_time * 0.5)
    remaining_time = deadline - time.time()
    if remaining_time > 3:
        routes = _or_opt(routes, instance, D, stations_set, deadline)

    # Final validity check
    if not _routes_valid(routes, instance, D, stations_set):
        routes = [[v] for v in range(1, instance["n_customers"] + 1)]
    return routes
# EVOLVE-BLOCK-END
