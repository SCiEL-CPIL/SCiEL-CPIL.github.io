# EVOLVE-BLOCK-START
import math
import time
import heapq
import random


def plan(instance):
    t_start = time.time()
    DEADLINE = 125.0

    def elapsed():
        return time.time() - t_start

    def finished():
        return elapsed() > DEADLINE

    # ---- Parse instance ----
    coords = instance["coords"]
    n_nodes = len(coords)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = list(instance["stations"])
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    customers = list(range(1, n_customers + 1))

    # ---- Precompute distances ----
    dist = [[0.0] * n_nodes for _ in range(n_nodes)]
    for i in range(n_nodes):
        xi, yi = coords[i]
        di = dist[i]
        for j in range(n_nodes):
            if j == i:
                continue
            dx = xi - coords[j][0]
            dy = yi - coords[j][1]
            di[j] = math.hypot(dx, dy)

    # ---- Feasibility of a route (returns total distance or None) ----
    def simulate(route):
        cur = 0
        t = 0.0
        b = float(battery_cap)
        rem_cap = float(capacity)
        total = 0.0
        for node in route:
            d = dist[cur][node]
            if b < d - 1e-9:
                return None
            b -= d
            t += d / speed
            total += d
            cur = node
            if node in customers:
                if rem_cap < demands[node] - 1e-9:
                    return None
                rem_cap -= demands[node]
                if t < ready[node]:
                    t = ready[node]
                if t > due[node] + 1e-9:
                    return None
                t += service[node]
        d = dist[cur][0]
        if b < d - 1e-9:
            return None
        b -= d
        t += d / speed
        total += d
        if t > horizon + 1e-9:
            return None
        return total

    # ---- Charging repair: try inserting one station to fix battery ----
    def repair_charging(route):
        cur = 0
        t = 0.0
        b = float(battery_cap)
        rem_cap = float(capacity)
        total = 0.0
        for idx, node in enumerate(route):
            d = dist[cur][node]
            if b < d - 1e-9:
                best = None
                for st in stations:
                    d1 = dist[cur][st]
                    if b < d1 - 1e-9:
                        continue
                    d2 = dist[st][node]
                    if d2 > battery_cap - 1e-9:
                        continue
                    extra = d1 + d2 - d
                    if best is None or extra < best[0]:
                        best = (extra, st)
                if best is None:
                    return None
                extra, st = best
                d1 = dist[cur][st]
                b -= d1
                t += d1 / speed
                total += d1
                charge = (battery_cap - b) * recharge_rate
                t += charge
                b = float(battery_cap)
                d2 = dist[st][node]
                b -= d2
                t += d2 / speed
                total += d2
                cur = node
            else:
                b -= d
                t += d / speed
                total += d
                cur = node
            if node in customers:
                if rem_cap < demands[node] - 1e-9:
                    return None
                rem_cap -= demands[node]
                if t < ready[node]:
                    t = ready[node]
                if t > due[node] + 1e-9:
                    return None
                t += service[node]
        d = dist[cur][0]
        if b < d - 1e-9:
            best = None
            for st in stations:
                d1 = dist[cur][st]
                if b < d1 - 1e-9:
                    continue
                d2 = dist[st][0]
                if d2 > battery_cap - 1e-9:
                    continue
                extra = d1 + d2 - d
                if best is None or extra < best[0]:
                    best = (extra, st)
            if best is None:
                return None
            extra, st = best
            d1 = dist[cur][st]
            b -= d1
            t += d1 / speed
            total += d1
            charge = (battery_cap - b) * recharge_rate
            t += charge
            b = float(battery_cap)
            d2 = dist[st][0]
            b -= d2
            t += d2 / speed
            total += d2
            cur = 0
        else:
            b -= d
            t += d / speed
            total += d
            cur = 0
        if t > horizon + 1e-9:
            return None
        return total

    # ---- Nearest-neighbour construction ----
    def construct(remaining, max_demand):
        route = []
        cur = 0
        t = 0.0
        b = float(battery_cap)
        rem_cap = float(max_demand)
        rem = set(remaining)
        while rem:
            best = None
            for c in rem:
                d = dist[cur][c]
                if b < d - 1e-9:
                    continue
                if rem_cap < demands[c] - 1e-9:
                    continue
                nt = t + d / speed
                if nt < ready[c]:
                    nt = ready[c]
                if nt > due[c] + 1e-9:
                    continue
                nt += service[c]
                if best is None or d < best[1]:
                    best = (c, d)
            if best is None:
                break
            c, d = best
            b -= d
            t += d / speed
            if t < ready[c]:
                t = ready[c]
            t += service[c]
            rem_cap -= demands[c]
            route.append(c)
            cur = c
            rem.discard(c)
        return route

    # ---- 2-opt on a single route (returns improved route or None if infeasible) ----
    def two_opt(route):
        n = len(route)
        if n < 3:
            return route
        improved = True
        best_route = list(route)
        best_val = simulate(best_route)
        if best_val is None:
            best_val = repair_charging(best_route)
        if best_val is None:
            return None
        iters = 0
        while improved and not finished() and iters < 40:
            improved = False
            iters += 1
            for i in range(n - 1):
                for j in range(i + 1, n):
                    new_route = best_route[:i] + best_route[i:j + 1][::-1] + best_route[j + 1:]
                    v = simulate(new_route)
                    if v is None:
                        v = repair_charging(new_route)
                    if v is not None and v < best_val - 1e-6:
                        best_route = new_route
                        best_val = v
                        improved = True
        return best_route

    # ---- k-means clustering ----
    def kmeans(k, seed):
        rng = random.Random(seed)
        idx = list(customers)
        if k >= len(idx):
            return [[c] for c in idx]
        # init: pick k spread-out points
        centers = [idx[0]]
        for _ in range(1, k):
            best_c = None
            best_score = -1
            for c in idx:
                if c in centers:
                    continue
                min_d = min(math.hypot(coords[c][0] - coords[cc][0],
                                       coords[c][1] - coords[cc][1]) for cc in centers)
                if min_d > best_score:
                    best_score = min_d
                    best_c = c
            if best_c is None:
                break
            centers.append(best_c)
        k = len(centers)
        assign = [0] * n_nodes
        for _ in range(50):
            for c in customers:
                bd = None
                bi = 0
                for i, cc in enumerate(centers):
                    d = math.hypot(coords[c][0] - coords[cc][0],
                                   coords[c][1] - coords[cc][1])
                    if bd is None or d < bd:
                        bd = d
                        bi = i
                assign[c] = bi
            new_centers = []
            for i in range(k):
                grp = [c for c in customers if assign[c] == i]
                if not grp:
                    new_centers.append(centers[i])
                    continue
                mx = sum(coords[c][0] for c in grp) / len(grp)
                my = sum(coords[c][1] for c in grp) / len(grp)
                # snap to nearest customer
                bc = grp[0]
                bdd = None
                for c in grp:
                    d = math.hypot(coords[c][0] - mx, coords[c][1] - my)
                    if bdd is None or d < bdd:
                        bdd = d
                        bc = c
                new_centers.append(bc)
            if new_centers == centers:
                break
            centers = new_centers
        groups = [[] for _ in range(k)]
        for c in customers:
            groups[assign[c]].append(c)
        return [g for g in groups if g]

    # ---- Balance a cluster set by moving customers ----
    def balance(groups, max_demand):
        groups = [list(g) for g in groups]
        # drop empty
        groups = [g for g in groups if g]
        changed = True
        iters = 0
        while changed and not finished() and iters < 30:
            changed = False
            iters += 1
            # find most overloaded group
            for gi in range(len(groups)):
                dem = sum(demands[c] for c in groups[gi])
                if dem <= max_demand:
                    continue
                # move customers from gi to least-loaded group
                for c in list(groups[gi]):
                    dem = sum(demands[c2] for c2 in groups[gi])
                    if dem <= max_demand:
                        break
                    # target: group with smallest resulting demand that can accept
                    best_j = None
                    best_dem = None
                    for gj in range(len(groups)):
                        if gj == gi:
                            continue
                        jd = sum(demands[c2] for c2 in groups[gj]) + demands[c]
                        if jd > max_demand:
                            continue
                        if best_dem is None or jd < best_dem:
                            best_dem = jd
                            best_j = gj
                    if best_j is None:
                        continue
                    groups[gi].remove(c)
                    groups[best_j].append(c)
                    changed = True
        # remove empties again
        return [g for g in groups if g]

    # ---- Build a full solution from clusters ----
    def build_solution(groups, seed):
        routes = []
        for g in groups:
            route = construct(g, capacity)
            if route:
                routes.append(route)
        return routes

    # ---- Local search across routes ----
    def local_search(routes):
        routes = [list(r) for r in routes]
        total = 0.0
        for r in routes:
            v = simulate(r)
            if v is None:
                v = repair_charging(r)
            if v is None:
                return None
            total += v
        improved = True
        iters = 0
        while improved and not finished() and iters < 60:
            improved = False
            iters += 1
            # 2-opt each route
            for i in range(len(routes)):
                nr = two_opt(routes[i])
                if nr is not None and nr != routes[i]:
                    routes[i] = nr
                    improved = True
            # relocate
            for i in range(len(routes)):
                for j in range(len(routes)):
                    if i == j:
                        continue
                    for pos in range(len(routes[i])):
                        c = routes[i][pos]
                        cand = routes[i][:pos] + routes[i][pos + 1:]
                        # try inserting c into j at each position
                        best = None
                        for p in range(len(routes[j]) + 1):
                            nrj = routes[j][:p] + [c] + routes[j][p:]
                            # evaluate both routes
                            vi = simulate(cand)
                            if vi is None:
                                vi = repair_charging(cand)
                            if vi is None:
                                continue
                            vj = simulate(nrj)
                            if vj is None:
                                vj = repair_charging(nrj)
                            if vj is None:
                                continue
                            val = vi + vj
                            if best is None or val < best[0]:
                                best = (val, p, cand, nrj)
                        if best is not None:
                            val, p, cand, nrj = best
                            # compare to current
                            cur_i = simulate(routes[i])
                            if cur_i is None:
                                cur_i = repair_charging(routes[i])
                            cur_j = simulate(routes[j])
                            if cur_j is None:
                                cur_j = repair_charging(routes[j])
                            if cur_i is not None and cur_j is not None and val < cur_i + cur_j - 1e-6:
                                routes[i] = cand
                                routes[j] = nrj
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # swap
            for i in range(len(routes)):
                for j in range(i + 1, len(routes)):
                    for pi in range(len(routes[i])):
                        for pj in range(len(routes[j])):
                            a = routes[i][pi]
                            b = routes[j][pj]
                            nr_i = list(routes[i])
                            nr_j = list(routes[j])
                            nr_i[pi] = b
                            nr_j[pj] = a
                            vi = simulate(nr_i)
                            if vi is None:
                                vi = repair_charging(nr_i)
                            vj = simulate(nr_j)
                            if vj is None:
                                vj = repair_charging(nr_j)
                            if vi is None or vj is None:
                                continue
                            cur_i = simulate(routes[i])
                            if cur_i is None:
                                cur_i = repair_charging(routes[i])
                            cur_j = simulate(routes[j])
                            if cur_j is None:
                                cur_j = repair_charging(routes[j])
                            if cur_i is not None and cur_j is not None and vi + vj < cur_i + cur_j - 1e-6:
                                routes[i] = nr_i
                                routes[j] = nr_j
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        # final check
        for r in routes:
            v = simulate(r)
            if v is None:
                v = repair_charging(r)
            if v is None:
                return None
        return routes

    # ---- Try multiple clusterings and keep best ----
    def total_of(routes):
        t = 0.0
        for r in routes:
            v = simulate(r)
            if v is None:
                v = repair_charging(r)
            if v is None:
                return None
            t += v
        return t

    best_sol = None
    best_val = float("inf")

    # Baseline: one customer per route (always feasible)
    baseline = [[c] for c in customers]
    bv = total_of(baseline)
    if bv is not None:
        best_sol = baseline
        best_val = bv

    # Try cluster-based solutions
    for k in [4, 5, 3, 6]:
        if finished():
            break
        for seed in [0, 1, 2]:
            if finished():
                break
            groups = kmeans(k, seed)
            groups = balance(groups, capacity)
            if not groups:
                continue
            routes = build_solution(groups, seed)
            if not routes:
                continue
            # ensure all customers covered; if not, add missing as singletons
            covered = set()
            for r in routes:
                covered.update(r)
            missing = [c for c in customers if c not in covered]
            for c in missing:
                routes.append([c])
            # local search
            ls = local_search(routes)
            if ls is not None:
                tv = total_of(ls)
                if tv is not None and tv < best_val:
                    best_val = tv
                    best_sol = ls

    if best_sol is None:
        best_sol = baseline
    return best_sol
# EVOLVE-BLOCK-END
