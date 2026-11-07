"""EV routing: savings-based construction + battery-aware local search."""
# EVOLVE-BLOCK-START
import numpy as np
from math import sqrt
import time
import random

DEADLINE_SECONDS = 142.0


def _build_dist(coords):
    n = len(coords)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            v = sqrt((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2)
            d[i, j] = v
            d[j, i] = v
    return d


def _sim(route, dist, inst):
    """Simulate route. Returns (feasible, total_dist, end_time, end_battery)."""
    nc = inst["n_customers"]
    bat_max = inst["battery"]
    spd = inst["speed"]
    rr = inst["recharge_rate"]
    hor = inst["horizon"]
    cap = inst["capacity"]
    dem = inst["demands"]
    ready = inst["ready"]
    due = inst["due"]
    svc = inst["service"]

    load = 0
    for nd in route:
        if nd <= nc:
            load += dem[nd]
    if load > cap:
        return (False, 0.0, 0.0, 0.0)

    bat = bat_max
    t = 0.0
    d_total = 0.0
    cur = 0

    for nd in route:
        dd = dist[cur, nd]
        t += dd / spd
        bat -= dd
        d_total += dd
        if bat < -1e-9:
            return (False, 0.0, 0.0, 0.0)
        if nd <= nc:
            t = max(t, ready[nd])
            if t > due[nd] + 1e-9:
                return (False, 0.0, 0.0, 0.0)
            t += svc[nd]
        else:
            need = bat_max - bat
            t += need * rr
            bat = bat_max
        cur = nd

    dd = dist[cur, 0]
    t += dd / spd
    bat -= dd
    d_total += dd
    if bat < -1e-9:
        return (False, 0.0, 0.0, 0.0)
    if t > hor + 1e-9:
        return (False, 0.0, 0.0, 0.0)
    return (True, d_total, t, bat)


def _validate(routes, inst, dist):
    nc = inst["n_customers"]
    served = set()
    total_d = 0.0
    for r in routes:
        ok, d, _, _ = _sim(r, dist, inst)
        if not ok:
            return (False, 0.0)
        total_d += d
        for nd in r:
            if nd <= nc:
                if nd in served:
                    return (False, 0.0)
                served.add(nd)
    if served != set(range(1, nc + 1)):
        return (False, 0.0)
    return (True, total_d)


def _total_dist(routes, dist):
    t = 0.0
    for r in routes:
        cur = 0
        for nd in r:
            t += dist[cur, nd]
            cur = nd
        t += dist[cur, 0]
    return t


def _route_load(route, inst):
    nc = inst["n_customers"]
    dem = inst["demands"]
    return sum(dem[nd] for nd in route if nd <= nc)


def _savings_construct(inst, dist, rng):
    """Clarke-Wright savings with battery/time feasibility checks."""
    nc = inst["n_customers"]
    cap = inst["capacity"]
    bat_max = inst["battery"]
    spd = inst["speed"]
    rr = inst["recharge_rate"]
    hor = inst["horizon"]
    dem = inst["demands"]
    ready = inst["ready"]
    due = inst["due"]
    svc = inst["service"]
    stations = set(inst["stations"])

    customers = list(range(1, nc + 1))
    # Sort by due time for better initial ordering
    customers.sort(key=lambda c: due[c])

    # Start with individual routes
    routes = [[c] for c in customers]
    loads = [dem[c] for c in customers]

    # Compute savings: s(i,j) = d(0,i) + d(0,j) - d(i,j)
    # Only consider merging if both are "end" of their routes
    # We'll do iterative savings merges

    n_r = len(routes)
    # Build savings list
    savings = []
    for i in range(n_r):
        for j in range(i + 1, n_r):
            a, b = routes[i][0], routes[i][-1]
            c, d = routes[j][0], routes[j][-1]
            # 4 possible merges
            s1 = dist[0, a] + dist[0, c] - dist[a, c]
            s2 = dist[0, a] + dist[0, d] - dist[a, d]
            s3 = dist[0, b] + dist[0, c] - dist[b, c]
            s4 = dist[0, b] + dist[0, d] - dist[b, d]
            # Store best saving merge
            best_s = max(s1, s2, s3, s4)
            if best_s > 0:
                # Determine which merge
                if best_s == s1:
                    savings.append((best_s, i, j, 0, 0))  # a-c merge
                elif best_s == s2:
                    savings.append((best_s, i, j, 0, 1))  # a-d merge
                elif best_s == s3:
                    savings.append((best_s, i, j, 1, 0))  # b-c merge
                else:
                    savings.append((best_s, i, j, 1, 1))  # b-d merge

    savings.sort(key=lambda x: -x[0])

    # Track which routes are still active
    active = list(range(n_r))
    route_map = {i: routes[i] for i in range(n_r)}
    load_map = {i: loads[i] for i in range(n_r)}

    # We need to handle the merge logic carefully
    # Let's use a different approach: maintain a list of routes and do pairwise merges

    # Reset: use a simpler iterative approach
    routes = [[c] for c in customers]
    loads = [dem[c] for c in customers]

    changed = True
    while changed:
        changed = False
        n_r = len(routes)
        # Compute all pairwise savings
        best_save = -1
        best_i = -1
        best_j = -1
        best_type = 0

        for i in range(n_r):
            for j in range(i + 1, n_r):
                if loads[i] + loads[j] > cap:
                    continue
                a, b = routes[i][0], routes[i][-1]
                c, d = routes[j][0], routes[j][-1]
                # Try 4 merges
                for typ in range(4):
                    if typ == 0:  # i + j (a...b + c...d -> a...b c...d)
                        merged = routes[i] + routes[j]
                        s = dist[0, a] + dist[0, c] - dist[b, c]
                    elif typ == 1:  # i + reversed j
                        merged = routes[i] + routes[j][::-1]
                        s = dist[0, a] + dist[0, d] - dist[b, d]
                    elif typ == 2:  # reversed i + j
                        merged = routes[i][::-1] + routes[j]
                        s = dist[0, b] + dist[0, c] - dist[a, c]
                    else:  # reversed i + reversed j
                        merged = routes[i][::-1] + routes[j][::-1]
                        s = dist[0, b] + dist[0, d] - dist[a, d]

                    if s <= best_save:
                        continue

                    # Check feasibility with station insertion
                    if _check_with_stations(merged, dist, inst):
                        best_save = s
                        best_i = i
                        best_j = j
                        best_type = typ

        if best_i >= 0:
            # Perform merge
            i, j, typ = best_i, best_j, best_type
            ri, rj = routes[i], routes[j]
            if typ == 0:
                merged = ri + rj
            elif typ == 1:
                merged = ri + rj[::-1]
            elif typ == 2:
                merged = ri[::-1] + rj
            else:
                merged = ri[::-1] + rj[::-1]

            new_load = loads[i] + loads[j]
            # Remove j (higher index) first
            routes.pop(j)
            loads.pop(j)
            routes[i] = merged
            loads[i] = new_load
            changed = True

    return routes


def _check_with_stations(route, dist, inst):
    """Check if route is feasible, allowing station insertion if needed."""
    nc = inst["n_customers"]
    bat_max = inst["battery"]
    spd = inst["speed"]
    rr = inst["recharge_rate"]
    hor = inst["horizon"]
    cap = inst["capacity"]
    dem = inst["demands"]
    ready = inst["ready"]
    due = inst["due"]
    svc = inst["service"]
    stations = set(inst["stations"])

    load = sum(dem[nd] for nd in route if nd <= nc)
    if load > cap:
        return False

    # Simulate and track battery
    bat = bat_max
    t = 0.0
    d_total = 0.0
    cur = 0
    new_route = []

    for nd in route:
        dd = dist[cur, nd]
        t += dd / spd
        bat -= dd
        d_total += dd

        if bat < -1e-9:
            # Need to insert a station before this node
            # Find best station to visit
            best_station = None
            best_s_dist = float('inf')
            for st in stations:
                d_to_st = dist[cur, st]
                d_st_to_nd = dist[st, nd]
                if bat + d_to_st < -1e-9:
                    continue
                # After charging, can we reach nd?
                if d_st_to_nd > bat_max + 1e-9:
                    continue
                total_d = d_to_st + d_st_to_nd
                if total_d < best_s_dist:
                    best_s_dist = total_d
                    best_station = st

            if best_station is None:
                return False

            # Insert station
            d_to_st = dist[cur, best_station]
            t += d_to_st / spd
            bat -= d_to_st
            d_total += d_to_st

            # Charge
            need = bat_max - bat
            t += need * rr
            bat = bat_max

            # Now go to nd
            d_st_to_nd = dist[best_station, nd]
            t += d_st_to_nd / spd
            bat -= d_st_to_nd
            d_total += d_st_to_nd

            if bat < -1e-9:
                return False

            new_route.append(best_station)

        if nd <= nc:
            t = max(t, ready[nd])
            if t > due[nd] + 1e-9:
                return False
            t += svc[nd]
        else:
            need = bat_max - bat
            t += need * rr
            bat = bat_max

        new_route.append(nd)
        cur = nd

    # Return to depot
    dd = dist[cur, 0]
    t += dd / spd
    bat -= dd
    d_total += dd

    if bat < -1e-9:
        # Try inserting station before depot
        best_station = None
        best_s_dist = float('inf')
        for st in stations:
            d_to_st = dist[cur, st]
            d_st_to_0 = dist[st, 0]
            if bat + d_to_st < -1e-9:
                continue
            if d_st_to_0 > bat_max + 1e-9:
                continue
            total_d = d_to_st + d_st_to_0
            if total_d < best_s_dist:
                best_s_dist = total_d
                best_station = st

        if best_station is None:
            return False

        d_to_st = dist[cur, best_station]
        t += d_to_st / spd
        bat -= d_to_st
        d_total += d_to_st
        need = bat_max - bat
        t += need * rr
        bat = bat_max
        d_st_to_0 = dist[best_station, 0]
        t += d_st_to_0 / spd
        bat -= d_st_to_0
        d_total += d_st_to_0

        if bat < -1e-9:
            return False

    if t > hor + 1e-9:
        return False

    return True


def _local_search(routes, inst, dist, rng, max_time=100.0):
    """2-opt + OR-opt local search with battery awareness."""
    nc = inst["n_customers"]
    bat_max = inst["battery"]
    spd = inst["speed"]
    rr = inst["recharge_rate"]
    hor = inst["horizon"]
    cap = inst["capacity"]
    dem = inst["demands"]
    stations = set(inst["stations"])

    start = time.time()
    cur_routes = [r[:] for r in routes]
    cur_dist = _total_dist(cur_routes, dist)
    best_routes = [r[:] for r in cur_routes]
    best_dist = cur_dist

    T = 30.0
    T_min = 0.01
    cool = 0.9997
    no_improve = 0
    max_no_improve = 5000

    for it in range(500000):
        if time.time() - start > max_time:
            break
        if no_improve > max_no_improve:
            T = max(T, 5.0)
            no_improve = 0

        T *= cool
        if T < T_min:
            T = T_min

        n_r = len(cur_routes)
        mt = rng.random()

        if mt < 0.5 and n_r >= 2:
            # OR-opt: move 1-3 nodes from one route to another
            r1 = rng.randint(0, n_r - 1)
            if len(cur_routes[r1]) < 2:
                continue
            ln = rng.choice([1, 2, 3])
            if ln > len(cur_routes[r1]) - 1:
                ln = max(1, len(cur_routes[r1]) - 1)
            p1 = rng.randint(0, len(cur_routes[r1]) - ln)

            r2 = rng.randint(0, n_r - 1)
            p2 = rng.randint(0, len(cur_routes[r2]) + 1)

            seq = cur_routes[r1][p1:p1 + ln]
            new_r1 = cur_routes[r1][:p1] + cur_routes[r1][p1 + ln:]
            new_r2 = cur_routes[r2][:p2] + seq + cur_routes[r2][p2:]

            # Check capacity
            l1 = _route_load(new_r1, inst)
            l2 = _route_load(new_r2, inst)
            if l1 > cap or l2 > cap:
                continue

            new_routes = [r[:] for r in cur_routes]
            new_routes[r1] = new_r1
            new_routes[r2] = new_r2
            new_routes = [r for r in new_routes if r]
            if len(new_routes) != n_r:
                continue

            ok, nd = _validate(new_routes, inst, dist)
            if not ok:
                continue

            delta = nd - cur_dist
            if delta < 0 or rng.random() < np.exp(-delta / T):
                cur_routes = new_routes
                cur_dist = nd
                if cur_dist < best_dist:
                    best_dist = cur_dist
                    best_routes = [r[:] for r in cur_routes]
                    no_improve = 0
                else:
                    no_improve += 1

        elif mt < 0.85:
            # 2-opt within a route
            ri = rng.randint(0, n_r - 1)
            if len(cur_routes[ri]) < 3:
                continue
            i = rng.randint(0, len(cur_routes[ri]) - 2)
            j = rng.randint(i + 1, len(cur_routes[ri]) - 1)

            new_route = cur_routes[ri][:i] + cur_routes[ri][i:j + 1][::-1] + cur_routes[ri][j + 1:]
            new_routes = [r[:] for r in cur_routes]
            new_routes[ri] = new_route

            ok, nd = _validate(new_routes, inst, dist)
            if not ok:
                continue

            delta = nd - cur_dist
            if delta < 0 or rng.random() < np.exp(-delta / T):
                cur_routes = new_routes
                cur_dist = nd
                if cur_dist < best_dist:
                    best_dist = cur_dist
                    best_routes = [r[:] for r in cur_routes]
                    no_improve = 0
                else:
                    no_improve += 1

        else:
            # Swap customers between routes
            if n_r < 2:
                continue
            r1 = rng.randint(0, n_r - 1)
            r2 = rng.randint(0, n_r - 1)
            if r1 == r2:
                continue
            if not cur_routes[r1] or not cur_routes[r2]:
                continue
            p1 = rng.randint(0, len(cur_routes[r1]) - 1)
            p2 = rng.randint(0, len(cur_routes[r2]) - 1)
            c1 = cur_routes[r1][p1]
            c2 = cur_routes[r2][p2]
            if c1 > nc or c2 > nc:
                continue

            l1 = _route_load(cur_routes[r1], inst) - dem[c1] + dem[c2]
            l2 = _route_load(cur_routes[r2], inst) - dem[c2] + dem[c1]
            if l1 > cap or l2 > cap:
                continue

            new_routes = [r[:] for r in cur_routes]
            new_routes[r1][p1] = c2
            new_routes[r2][p2] = c1

            ok, nd = _validate(new_routes, inst, dist)
            if not ok:
                continue

            delta = nd - cur_dist
            if delta < 0 or rng.random() < np.exp(-delta / T):
                cur_routes = new_routes
                cur_dist = nd
                if cur_dist < best_dist:
                    best_dist = cur_dist
                    best_routes = [r[:] for r in cur_routes]
                    no_improve = 0
                else:
                    no_improve += 1

    return best_routes


def _insert_stations(routes, inst, dist):
    """Insert charging stations where needed for battery feasibility."""
    nc = inst["n_customers"]
    bat_max = inst["battery"]
    spd = inst["speed"]
    rr = inst["recharge_rate"]
    hor = inst["horizon"]
    stations = set(inst["stations"])

    new_routes = []
    for route in routes:
        # Simulate and find where battery drops below 0
        bat = bat_max
        t = 0.0
        cur = 0
        new_route = []
        needs_station = False

        for nd in route:
            dd = dist[cur, nd]
            bat_after = bat - dd
            if bat_after < -1e-9:
                # Need station before this node
                best_st = None
                best_d = float('inf')
                for st in stations:
                    d1 = dist[cur, st]
                    d2 = dist[st, nd]
                    if bat - d1 < -1e-9:
                        continue
                    if d2 > bat_max + 1e-9:
                        continue
                    tot = d1 + d2
                    if tot < best_d:
                        best_d = tot
                        best_st = st
                if best_st is None:
                    # Can't fix this route
                    new_routes.append(route)
                    needs_station = True
                    break
                # Insert station
                new_route.append(best_st)
                bat = bat_max  # after charging
                cur = best_st
                # Now go to nd
                dd2 = dist[cur, nd]
                bat -= dd2
                t += dd2 / spd
                cur = nd
            else:
                bat = bat_after
                t += dd / spd
                cur = nd

            if nd <= nc:
                t = max(t, inst["ready"][nd])
                t += inst["service"][nd]
            else:
                need = bat_max - bat
                t += need * rr
                bat = bat_max

            new_route.append(nd)

        if not needs_station:
            # Check return to depot
            dd = dist[cur, 0]
            bat_final = bat - dd
            if bat_final < -1e-9:
                # Need station before depot
                best_st = None
                best_d = float('inf')
                for st in stations:
                    d1 = dist[cur, st]
                    d2 = dist[st, 0]
                    if bat - d1 < -1e-9:
                        continue
                    if d2 > bat_max + 1e-9:
                        continue
                    tot = d1 + d2
                    if tot < best_d:
                        best_d = tot
                        best_st = st
                if best_st is not None:
                    new_route.append(best_st)
            new_routes.append(new_route)

    return new_routes


def plan(instance):
    start = time.time()
    rng = random.Random(42)

    coords = np.array(instance["coords"], dtype=np.float64)
    dist = _build_dist(coords)

    # Savings-based construction
    routes = _savings_construct(instance, dist, rng)

    # Insert stations if needed
    routes = _insert_stations(routes, instance, dist)

    # Validate
    ok, d = _validate(routes, instance, dist)
    if not ok:
        # Fallback: individual routes with stations
        nc = instance["n_customers"]
        routes = [[c] for c in range(1, nc + 1)]
        routes = _insert_stations(routes, instance, dist)
        ok, d = _validate(routes, instance, dist)
        if not ok:
            # Last resort
            return [[c] for c in range(1, nc + 1)]

    # Local search
    remaining = DEADLINE_SECONDS - (time.time() - start) - 3
    if remaining > 5:
        routes = _local_search(routes, instance, dist, rng, max_time=remaining)

    # Final validation
    ok, _ = _validate(routes, instance, dist)
    if not ok:
        nc = instance["n_customers"]
        routes = [[c] for c in range(1, nc + 1)]

    return routes
# EVOLVE-BLOCK-END
