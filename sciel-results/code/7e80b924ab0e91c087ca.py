# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def plan(instance):
    t0 = time.time()
    deadline = t0 + 145.0

    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_customers = int(instance["n_customers"])
    stations = list(instance["stations"])
    capacity = float(instance["capacity"])
    battery_cap = float(instance["battery"])
    speed = float(instance["speed"])
    recharge_rate = float(instance["recharge_rate"])
    horizon = float(instance["horizon"])

    n_nodes = len(coords)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff * diff).sum(-1))

    customers = list(range(1, n_customers + 1))
    cust_set = set(customers)
    station_set = set(stations)
    is_customer = np.zeros(n_nodes, dtype=bool)
    is_station = np.zeros(n_nodes, dtype=bool)
    for c in customers:
        is_customer[c] = True
    for s in stations:
        is_station[s] = True

    cust_demand = np.zeros(n_nodes)
    for c in customers:
        cust_demand[c] = demands[c]

    # Precompute station insertion cost for each pair (prev, next)
    # For each position in a route (between prev and next), what's the best station to insert?
    # This is expensive to precompute fully, so we compute on the fly but cache.

    def simulate(route):
        """Simulate a route. Returns (total_dist, total_time, final_battery) or None if infeasible."""
        if not route:
            return (0.0, 0.0, battery_cap)
        t = 0.0
        bat = battery_cap
        total_d = 0.0
        prev = 0
        for node in route:
            d = dist[prev, node]
            t += d / speed
            bat -= d
            if bat < -1e-9:
                return None
            total_d += d
            if is_customer[node]:
                if t < ready[node]:
                    t = ready[node]
                if t > due[node] + 1e-9:
                    return None
                t += service[node]
            elif is_station[node]:
                t += (battery_cap - bat) * recharge_rate
                bat = battery_cap
            prev = node
        d = dist[prev, 0]
        t += d / speed
        bat -= d
        total_d += d
        if bat < -1e-9 or t > horizon + 1e-9:
            return None
        return (total_d, t, bat)

    def route_demand(route):
        s = 0.0
        for v in route:
            if is_customer[v]:
                s += cust_demand[v]
        return s

    def total_dist(routes):
        tot = 0.0
        for r in routes:
            s = simulate(r)
            if s is None:
                return float("inf")
            tot += s[0]
        return tot

    def try_insert_station(route, target_pos, budget=200):
        """Try to insert a station at or near target_pos to make route feasible.
        Returns (fixed_route, dist) or (None, None)."""
        s = simulate(route)
        if s is not None:
            return list(route), s[0]

        # Find where battery goes negative
        t = 0.0
        bat = battery_cap
        cur = 0
        neg_pos = -1
        for idx, node in enumerate(route):
            d = dist[cur, node]
            t += d / speed
            bat -= d
            if bat < -1e-9 and neg_pos == -1:
                neg_pos = idx
            if is_customer[node]:
                if t < ready[node]:
                    t = ready[node]
                t += service[node]
            elif is_station[node]:
                t += (battery_cap - bat) * recharge_rate
                bat = battery_cap
            cur = node
        if neg_pos == -1:
            d = dist[cur, 0]
            bat -= d
            if bat < -1e-9:
                neg_pos = len(route)
            else:
                return None, None

        # Try inserting stations around the problematic position
        best_route = None
        best_d = float('inf')
        search_start = max(0, neg_pos - 3)
        search_end = min(len(route) + 1, neg_pos + 4)
        for p in range(search_start, search_end + 1):
            for st in stations:
                cand = list(route[:p]) + [st] + list(route[p:])
                sc = simulate(cand)
                if sc is not None:
                    if sc[0] < best_d:
                        best_route, best_d = cand, sc[0]
        if best_route is not None:
            return best_route, best_d

        # Broader search
        for p in range(len(route) + 1):
            for st in stations:
                cand = list(route[:p]) + [st] + list(route[p:])
                sc = simulate(cand)
                if sc is not None:
                    if sc[0] < best_d:
                        best_route, best_d = cand, sc[0]
        if best_route is not None:
            return best_route, best_d
        return None, None

    def greedy_construct(order):
        """Construct routes by inserting customers in the given order."""
        routes = []
        for c in order:
            placed = False
            best = None
            best_cd = float('inf')
            for ri, r in enumerate(routes):
                if route_demand(r) + demands[c] > capacity:
                    continue
                # Find best feasible position
                positions = []
                for pos in range(len(r) + 1):
                    prev = r[pos - 1] if pos > 0 else 0
                    nxt = r[pos] if pos < len(r) else 0
                    delta = dist[prev, c] + dist[c, nxt] - dist[prev, nxt]
                    positions.append((delta, pos))
                positions.sort()

                for delta, pos in positions:
                    cand = list(r[:pos]) + [c] + list(r[pos:])
                    fixed, cd = try_insert_station(cand)
                    if fixed is not None:
                        if cd < best_cd:
                            best = (ri, fixed, cd)
                            best_cd = cd
                        break
            if best is not None:
                routes[best[0]] = best[1]
                placed = True
            if not placed:
                fixed, cd = try_insert_station([c])
                if fixed is None:
                    fixed = [c]
                routes.append(fixed)
        return routes

    def two_opt(route, max_iter=300, tol=1e-9):
        """2-opt + or-opt on a single route."""
        if len(route) < 2:
            return route
        cur = simulate(route)
        if cur is None:
            return route
        cur_d = cur[0]
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            n = len(route)
            for i in range(n - 1):
                for j in range(i + 1, n):
                    a = route[i - 1] if i > 0 else 0
                    b = route[i]
                    c = route[j]
                    d = route[j + 1] if j + 1 < n else 0
                    old_len = dist[a, b] + dist[c, d]
                    new_len = dist[a, c] + dist[b, d]
                    if new_len >= old_len - tol:
                        continue
                    nr = list(route[:i]) + list(reversed(route[i:j + 1])) + list(route[j + 1:])
                    sn = simulate(nr)
                    if sn is None:
                        continue
                    if sn[0] < cur_d - tol:
                        route, cur_d = nr, sn[0]
                        improved = True
            # Or-opt: relocate blocks of length 1 and 2
            for blocklen in [1, 2]:
                n = len(route)
                if n < blocklen + 1:
                    continue
                for _ in range(50):
                    improved2 = False
                    for i in range(n - blocklen + 1):
                        block = route[i:i + blocklen]
                        rest = route[:i] + route[i + blocklen:]
                        for j in range(len(rest) + 1):
                            if j <= i:
                                continue
                            cand = rest[:j] + block + rest[j:]
                            if cand == route:
                                continue
                            sn = simulate(cand)
                            if sn is None:
                                continue
                            if sn[0] < cur_d - tol:
                                route, cur_d = cand, sn[0]
                                improved2 = True
                                improved = True
                    if not improved2:
                        break
        return route

    def relocate(routes, blocklen=1):
        """Move a block of size blocklen from one route to another."""
        nR = len(routes)
        oldD = [(simulate(r)[0] if r else 0.0) for r in routes]
        for i in range(nR):
            ri_full = routes[i]
            if not ri_full:
                continue
            old_di = oldD[i]
            for pi in range(len(ri_full) - blocklen + 1):
                block = ri_full[pi:pi + blocklen]
                block_demand = 0.0
                for v in block:
                    if is_customer[v]:
                        block_demand += cust_demand[v]
                rest_i = ri_full[:pi] + ri_full[pi + blocklen:]
                sim_i = simulate(rest_i) if rest_i else (0.0, 0.0, battery_cap)
                if sim_i is None:
                    continue
                di = sim_i[0] if rest_i else 0.0
                for j in range(nR):
                    if i == j:
                        continue
                    old_dj = oldD[j]
                    if route_demand(routes[j]) + block_demand > capacity:
                        continue
                    rj_base = routes[j]
                    best_pj = -1
                    best_delta = float('inf')
                    for pj in range(len(rj_base) + 1):
                        prev = rj_base[pj - 1] if pj > 0 else 0
                        nxt = rj_base[pj] if pj < len(rj_base) else 0
                        b0 = block[0]
                        b1 = block[-1]
                        delta = dist[prev, b0] + dist[b1, nxt] - dist[prev, nxt]
                        if delta < best_delta:
                            best_delta = delta
                            best_pj = pj
                    if best_pj == -1:
                        continue
                    rj = rj_base[:best_pj] + list(block) + rj_base[best_pj:]
                    sim_j = simulate(rj)
                    if sim_j is None:
                        continue
                    if di + sim_j[0] < old_di + old_dj - 1e-7:
                        routes[i] = rest_i
                        routes[j] = rj
                        routes = [r for r in routes if r]
                        return routes, True
        # Cross-route swap of single customers
        for i in range(nR):
            for j in range(i + 1, nR):
                for pi in range(len(routes[i])):
                    ci = routes[i][pi]
                    if not is_customer[ci]:
                        continue
                    for pj in range(len(routes[j])):
                        cj = routes[j][pj]
                        if not is_customer[cj]:
                            continue
                        if route_demand(routes[i]) - demands[ci] + demands[cj] > capacity:
                            continue
                        if route_demand(routes[j]) - demands[cj] + demands[ci] > capacity:
                            continue
                        ni = list(routes[i])
                        nj = list(routes[j])
                        ni[pi] = cj
                        nj[pj] = ci
                        si = simulate(ni)
                        sj = simulate(nj)
                        if si is None or sj is None:
                            continue
                        if si[0] + sj[0] < oldD[i] + oldD[j] - 1e-7:
                            routes[i] = ni
                            routes[j] = nj
                            return routes, True
        return routes, False

    def deep_search(routes, budget_end):
        """Iterative local search until no improvement or time budget exhausted."""
        cur = total_dist(routes)
        if cur == float("inf"):
            return routes, cur
        while time.time() < budget_end:
            improved = False
            # 2-opt on each route
            nr = [two_opt(r) for r in routes]
            nd = total_dist(nr)
            if nd < cur - 1e-9:
                routes, cur = nr, nd
                improved = True
            # Relocate blocks of size 1
            nr, imp = relocate(routes, 1)
            if imp:
                nd = total_dist(nr)
                if nd < cur - 1e-9:
                    routes, cur = nr, nd
                    improved = True
                    nr2 = [two_opt(r) for r in routes]
                    nd2 = total_dist(nr2)
                    if nd2 < cur - 1e-9:
                        routes, cur = nr2, nd2
            # Relocate blocks of size 2
            nr, imp = relocate(routes, 2)
            if imp:
                nd = total_dist(nr)
                if nd < cur - 1e-9:
                    routes, cur = nr, nd
                    improved = True
            if not improved:
                break
        return routes, cur

    # Build diverse candidates
    candidates = []
    order_angle = sorted(customers, key=lambda v: math.atan2(coords[v, 1] - coords[0, 1],
                                                            coords[v, 0] - coords[0, 0]))
    candidates.append(greedy_construct(order_angle))
    order_due = sorted(customers, key=lambda v: due[v])
    candidates.append(greedy_construct(order_due))
    order_dist = sorted(customers, key=lambda v: dist[0, v])
    candidates.append(greedy_construct(order_dist))

    rnd = random.Random(42)
    for _ in range(8):
        o = customers[:]
        rnd.shuffle(o)
        candidates.append(greedy_construct(o))

    scored = []
    for cand in candidates:
        if cand is None:
            continue
        c2 = [two_opt(r) for r in cand]
        d = total_dist(c2)
        if d < float("inf"):
            scored.append((d, [list(r) for r in c2]))
    scored.sort(key=lambda x: x[0])

    if scored:
        global_best, global_d = scored[0][1], scored[0][0]
    else:
        global_best, global_d = None, float("inf")

    # Deep search on top candidates
    for d0, cand in scored:
        if time.time() > deadline - 25.0:
            break
        r, dd = deep_search(cand, deadline - 20.0)
        if dd < global_d - 1e-9:
            global_best, global_d = r, dd

    # Final polish pass
    if global_best is not None and time.time() < deadline - 10.0:
        polish_end = deadline - 8.0
        cur_polish = total_dist(global_best)
        improved_polish = True
        while improved_polish and time.time() < polish_end:
            improved_polish = False
            nr = [two_opt(r, max_iter=400, tol=1e-10) for r in global_best]
            nd = total_dist(nr)
            if nd < cur_polish - 1e-10:
                global_best, cur_polish = nr, nd
                improved_polish = True
            nr, imp = relocate(global_best, 1)
            if imp:
                nd = total_dist(nr)
                if nd < cur_polish - 1e-10:
                    global_best, cur_polish = nr, nd
                    improved_polish = True
                    nr2 = [two_opt(r, max_iter=300, tol=1e-10) for r in global_best]
                    nd2 = total_dist(nr2)
                    if nd2 < cur_polish - 1e-10:
                        global_best, cur_polish = nr2, nd2
                        improved_polish = True
            nr, imp = relocate(global_best, 2)
            if imp:
                nd = total_dist(nr)
                if nd < cur_polish - 1e-10:
                    global_best, cur_polish = nr, nd
                    improved_polish = True

    # Additional random restarts
    extra = random.Random(999)
    k = 0
    while time.time() < deadline - 30.0 and k < 50:
        o = customers[:]
        extra.shuffle(o)
        cand = greedy_construct(o)
        if cand is None:
            k += 1
            continue
        cand = [two_opt(r) for r in cand]
        d = total_dist(cand)
        if d == float("inf"):
            k += 1
            continue
        r, dd = deep_search(cand, deadline - 25.0)
        if dd < global_d - 1e-9:
            global_best, global_d = r, dd
        k += 1

    routes = global_best

    def all_feasible(rs):
        served = []
        for r in rs:
            s = simulate(r)
            if s is None:
                return False
            for v in r:
                if is_customer[v]:
                    served.append(v)
        return len(served) == len(customers) and len(set(served)) == len(customers)

    if routes is None or not all_feasible(routes):
        routes = [[c] for c in customers]
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                if route_demand(routes[i]) + route_demand(routes[j]) > capacity:
                    continue
                cand = routes[i] + routes[j]
                fixed, cd = try_insert_station(cand)
                if fixed is not None:
                    routes[i] = fixed
                    routes.pop(j)
                    break

    return routes
# EVOLVE-BLOCK-END
