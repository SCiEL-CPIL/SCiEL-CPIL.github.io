# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random
import heapq


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

    # Precompute distance arrays for faster access
    dist_depot = dist[0, :]

    def simulate_fast(route):
        """Fast simulation. Returns (dist, time, battery) or None."""
        if not route:
            return (0.0, 0.0, battery_cap)
        t = 0.0
        bat = battery_cap
        cd = 0.0
        cur = 0
        for node in route:
            d = dist[cur, node]
            t += d / speed
            bat -= d
            if bat < -1e-9:
                return None
            cd += d
            if is_customer[node]:
                if t < ready[node]:
                    t = ready[node]
                if t > due[node] + 1e-9:
                    return None
                t += service[node]
            elif is_station[node]:
                t += (battery_cap - bat) * recharge_rate
                bat = battery_cap
            cur = node
        d = dist[cur, 0]
        t += d / speed
        bat -= d
        cd += d
        if bat < -1e-9 or t > horizon + 1e-9:
            return None
        return (cd, t, bat)

    def route_demand(route):
        s = 0.0
        for v in route:
            if is_customer[v]:
                s += cust_demand[v]
        return s

    def total_dist(routes):
        tot = 0.0
        for r in routes:
            s = simulate_fast(r)
            if s is None:
                return float("inf")
            tot += s[0]
        return tot

    def remove_stations(routes):
        """Remove any station that is not needed (battery feasible without it)."""
        changed = False
        for ri in range(len(routes)):
            r = routes[ri]
            if not r:
                continue
            new_r = []
            for node in r:
                if is_station[node]:
                    cand = new_r + [node] + r[r.index(node) + 1:]
                    # Actually build properly
                    idx = r.index(node)
                    cand = r[:idx] + r[idx+1:]
                    sc = simulate_fast(cand)
                    if sc is None:
                        new_r.append(node)
                    # else skip station
                else:
                    new_r.append(node)
            if new_r != r:
                routes[ri] = new_r
                changed = True
        return routes, changed

    def try_insert_station(route, pos, st):
        """Try inserting station st at position pos in route. Return (fixed_route, dist) or (None, None)."""
        cand = list(route[:pos]) + [st] + list(route[pos:])
        if route_demand(cand) > capacity:
            return None, None
        sc = simulate_fast(cand)
        if sc is not None:
            return cand, sc[0]
        return None, None

    def best_station_fix(route):
        """Insert cheapest station(s) if needed to make route feasible."""
        s = simulate_fast(route)
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

        # Try inserting stations near where battery runs low
        best_route = None
        best_d = float('inf')
        search_start = max(0, neg_pos - 3)
        search_end = min(len(route) + 1, neg_pos + 4)
        for p in range(search_start, search_end + 1):
            for st in stations:
                fixed, cd = try_insert_station(route, p, st)
                if fixed is not None and cd < best_d:
                    best_route, best_d = fixed, cd
        if best_route is not None:
            return best_route, best_d
        # Broader search
        for p in range(len(route) + 1):
            for st in stations:
                fixed, cd = try_insert_station(route, p, st)
                if fixed is not None and cd < best_d:
                    best_route, best_d = fixed, cd
        if best_route is not None:
            return best_route, best_d
        return None, None

    def greedy_construct(order):
        """Construct routes by inserting customers in given order."""
        routes = []
        for c in order:
            placed = False
            best = None
            for ri, r in enumerate(routes):
                if route_demand(r) + demands[c] > capacity:
                    continue
                best_pos = -1
                best_delta = float('inf')
                for pos in range(len(r) + 1):
                    prev = r[pos - 1] if pos > 0 else 0
                    nxt = r[pos] if pos < len(r) else 0
                    delta = dist[prev, c] + dist[c, nxt] - dist[prev, nxt]
                    if delta < best_delta:
                        best_delta = delta
                        best_pos = pos
                if best_pos == -1:
                    continue
                cand = list(r[:best_pos]) + [c] + list(r[best_pos:])
                fixed, cd = best_station_fix(cand)
                if fixed is None:
                    continue
                if best is None or cd < best[2]:
                    best = (ri, fixed, cd)
            if best is not None:
                routes[best[0]] = best[1]
                placed = True
            if not placed:
                fixed, cd = best_station_fix([c])
                if fixed is None:
                    fixed = [c]
                routes.append(fixed)
        return routes

    def nearest_neighbor_construct():
        """Nearest neighbor construction with battery awareness."""
        unvisited = set(customers)
        routes = []
        while unvisited:
            # Start new route from depot
            route = []
            cur = 0
            t = 0.0
            bat = battery_cap
            cd = 0.0
            while unvisited:
                # Find nearest unvisited customer that's feasible
                best_c = None
                best_score = float('inf')
                for c in unvisited:
                    d = dist[cur, c]
                    new_t = t + d / speed
                    if new_t < ready[c]:
                        new_t = ready[c]
                    if new_t > due[c] + 1e-9:
                        continue
                    new_bat = bat - d
                    if new_bat < -1e-9:
                        # Check if we can reach a station first
                        can_reach = False
                        for st in stations:
                            ds = dist[cur, st]
                            if bat - ds >= -1e-9:
                                can_reach = True
                                break
                        if not can_reach:
                            continue
                    # Score: distance + small penalty for time
                    score = d + 0.01 * new_t
                    if score < best_score:
                        best_score = score
                        best_c = c
                if best_c is None:
                    break
                unvisited.discard(best_c)
                d = dist[cur, best_c]
                t += d / speed
                if t < ready[best_c]:
                    t = ready[best_c]
                t += service[best_c]
                bat -= d
                cd += d
                route.append(best_c)
                cur = best_c
            # Try to fix battery issues
            fixed, fcd = best_station_fix(route)
            if fixed is not None:
                routes.append(fixed)
            elif route:
                routes.append(route)
        return routes

    def two_opt(route, max_iter=300):
        """2-opt and or-opt local search on a single route."""
        if len(route) < 2:
            return route
        cur = simulate_fast(route)
        if cur is None:
            return route
        cur_d = cur[0]
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            n = len(route)
            # 2-opt
            for i in range(n - 1):
                for j in range(i + 1, n):
                    a = route[i - 1] if i > 0 else 0
                    b = route[i]
                    c = route[j]
                    d = route[j + 1] if j + 1 < n else 0
                    old_len = dist[a, b] + dist[c, d]
                    new_len = dist[a, c] + dist[b, d]
                    if new_len >= old_len - 1e-7:
                        continue
                    nr = list(route[:i]) + list(reversed(route[i:j + 1])) + list(route[j + 1:])
                    if route_demand(nr) > capacity:
                        continue
                    sn = simulate_fast(nr)
                    if sn is None:
                        continue
                    if sn[0] < cur_d - 1e-7:
                        route, cur_d = nr, sn[0]
                        improved = True
            # Or-opt: relocate blocks of length 1, 2, 3
            for blocklen in [1, 2, 3]:
                n = len(route)
                if n < blocklen + 1:
                    continue
                for i in range(n - blocklen + 1):
                    block = route[i:i + blocklen]
                    rest = route[:i] + route[i + blocklen:]
                    for j in range(len(rest) + 1):
                        if j <= i and j > 0:
                            continue
                        if j == i:
                            continue
                        cand = rest[:j] + block + rest[j:]
                        if cand == route:
                            continue
                        sn = simulate_fast(cand)
                        if sn is None:
                            continue
                        if sn[0] < cur_d - 1e-7:
                            route, cur_d = cand, sn[0]
                            improved = True
        return route

    def savings_construct():
        """Clarke-Wright savings heuristic for initial construction."""
        # Start with each customer as its own route
        routes = [[c] for c in customers]
        # Compute savings: s(i,j) = d(0,i) + d(0,j) - d(i,j)
        savings = []
        for i in range(len(customers)):
            for j in range(i + 1, len(customers)):
                ci = customers[i]
                cj = customers[j]
                s = dist[0, ci] + dist[0, cj] - dist[ci, cj]
                savings.append((s, i, j))
        savings.sort(key=lambda x: -x[0])

        # Union-find for route membership
        parent = list(range(len(customers)))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        # Build routes by merging
        route_map = {i: [customers[i]] for i in range(len(customers))}
        # We need to track which route each customer is in and merge
        # Simpler: track routes as lists, merge when savings beneficial
        active = {i: [customers[i]] for i in range(len(customers))}
        cust_to_route = {customers[i]: i for i in range(len(customers))}

        for s, i, j in savings:
            if s <= 0:
                break
            ri = cust_to_route[customers[i]]
            rj = cust_to_route[customers[j]]
            if ri == rj:
                continue
            # Try to merge: customers[i] must be at end of route ri, customers[j] at start of route rj (or vice versa)
            ri_list = active[ri]
            rj_list = active[rj]
            ci, cj = customers[i], customers[j]
            merged = None
            # Option 1: ci at end of ri, cj at start of rj
            if ri_list[-1] == ci and rj_list[0] == cj:
                merged = ri_list + rj_list
            # Option 2: ci at start of ri, cj at end of rj
            elif ri_list[0] == ci and rj_list[-1] == cj:
                merged = list(reversed(ri_list)) + rj_list
            # Option 3: cj at end of rj, ci at start of ri
            elif rj_list[-1] == cj and ri_list[0] == ci:
                merged = rj_list + ri_list
            # Option 4: cj at start of rj, ci at end of ri
            elif rj_list[0] == cj and ri_list[-1] == ci:
                merged = list(reversed(rj_list)) + ri_list
            else:
                continue
            # Check feasibility
            if route_demand(merged) > capacity:
                continue
            fixed, cd = best_station_fix(merged)
            if fixed is None:
                continue
            # Merge
            active[ri] = fixed
            for c in rj_list:
                cust_to_route[c] = ri
            del active[rj]

        # Collect final routes
        result = [r for r in active.values() if r]
        return result

    def relocate(routes, blocklen=1):
        """Relocate blocks between routes."""
        nR = len(routes)
        if nR < 2:
            return routes, False
        oldD = [(simulate_fast(r)[0] if r else 0.0) for r in routes]
        for i in range(nR):
            ri_full = routes[i]
            old_di = oldD[i]
            for pi in range(len(ri_full) - blocklen + 1):
                block = ri_full[pi:pi + blocklen]
                block_demand = 0.0
                for v in block:
                    if is_customer[v]:
                        block_demand += cust_demand[v]
                rest_i = ri_full[:pi] + ri_full[pi + blocklen:]
                sim_i = simulate_fast(rest_i) if rest_i else (0.0, 0.0, battery_cap)
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
                    sim_j = simulate_fast(rj)
                    if sim_j is None:
                        continue
                    if di + sim_j[0] < old_di + old_dj - 1e-7:
                        routes[i] = rest_i
                        routes[j] = rj
                        routes = [r for r in routes if r]
                        return routes, True
        return routes, False

    def cross_swap(routes):
        """Swap customers between two routes."""
        nR = len(routes)
        if nR < 2:
            return routes, False
        oldD = [(simulate_fast(r)[0] if r else 0.0) for r in routes]
        for i in range(nR):
            old_di = oldD[i]
            for j in range(i + 1, nR):
                old_dj = oldD[j]
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
                        si = simulate_fast(ni)
                        sj = simulate_fast(nj)
                        if si is None or sj is None:
                            continue
                        if si[0] + sj[0] < old_di + old_dj - 1e-7:
                            routes[i] = ni
                            routes[j] = nj
                            return routes, True
        return routes, False

    def deep_search(routes, budget_end):
        """Iterative improvement until budget exhausted."""
        cur = total_dist(routes)
        if cur == float("inf"):
            return routes, cur
        while time.time() < budget_end:
            improved = False
            # 2-opt + Or-opt on each route
            nr = [two_opt(r) for r in routes]
            nd = total_dist(nr)
            if nd < cur - 1e-7:
                routes, cur = nr, nd
                improved = True
            # Remove unnecessary stations
            nr, imp = remove_stations(routes)
            if imp:
                nd = total_dist(nr)
                if nd < cur - 1e-7:
                    routes, cur = nr, nd
                    improved = True
            # Relocate blocks of size 1
            nr, imp = relocate(routes, 1)
            if imp:
                nd = total_dist(nr)
                if nd < cur - 1e-7:
                    routes, cur = nr, nd
                    improved = True
                    nr2 = [two_opt(r) for r in routes]
                    nd2 = total_dist(nr2)
                    if nd2 < cur - 1e-7:
                        routes, cur = nr2, nd2
            # Relocate blocks of size 2
            nr, imp = relocate(routes, 2)
            if imp:
                nd = total_dist(nr)
                if nd < cur - 1e-7:
                    routes, cur = nr, nd
                    improved = True
            # Cross-route swap
            nr, imp = cross_swap(routes)
            if imp:
                nd = total_dist(nr)
                if nd < cur - 1e-7:
                    routes, cur = nr, nd
                    improved = True
            if not improved:
                break
        return routes, cur

    # Build diverse candidates using multiple construction strategies
    candidates = []

    # Clarke-Wright savings
    try:
        candidates.append(savings_construct())
    except Exception:
        pass

    # Angle-based
    order_angle = sorted(customers, key=lambda v: math.atan2(coords[v, 1] - coords[0, 1],
                                                            coords[v, 0] - coords[0, 0]))
    candidates.append(greedy_construct(order_angle))

    # Due-date based
    order_due = sorted(customers, key=lambda v: due[v])
    candidates.append(greedy_construct(order_due))

    # Distance from depot
    order_dist = sorted(customers, key=lambda v: dist[0, v])
    candidates.append(greedy_construct(order_dist))

    # Nearest neighbor
    candidates.append(nearest_neighbor_construct())

    # Random orders
    rnd = random.Random(12345)
    for _ in range(6):
        o = customers[:]
        rnd.shuffle(o)
        candidates.append(greedy_construct(o))

    # Score and sort candidates
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

    # Deep search on top candidates, prioritizing better ones
    for d0, cand in scored:
        if time.time() > deadline - 10.0:
            break
        r, dd = deep_search(cand, deadline - 8.0)
        if dd < global_d - 1e-9:
            global_best, global_d = r, dd

    # Additional random restarts
    extra = random.Random(777)
    k = 0
    while time.time() < deadline - 15.0 and k < 30:
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
        r, dd = deep_search(cand, deadline - 10.0)
        if dd < global_d - 1e-9:
            global_best, global_d = r, dd
        k += 1

    routes = global_best

    def all_feasible(rs):
        served = []
        for r in rs:
            s = simulate_fast(r)
            if s is None:
                return False
            for v in r:
                if is_customer[v]:
                    served.append(v)
        return len(served) == len(customers) and len(set(served)) == len(customers)

    if routes is None or not all_feasible(routes):
        # Fallback: one customer per route
        routes = []
        for c in customers:
            fixed, _ = best_station_fix([c])
            if fixed is None:
                fixed = [c]
            routes.append(fixed)

    # Final polish: remove stations and do one more 2-opt pass
    if time.time() < deadline - 2.0:
        routes, _ = remove_stations(routes)
        routes = [two_opt(r) for r in routes]
        routes, _ = remove_stations(routes)

    return routes
# EVOLVE-BLOCK-END
