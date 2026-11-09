# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def plan(instance):
    # Strict wall-clock guard (evaluator kills at 150s; keep a safety margin).
    t0 = time.time()
    DEADLINE = t0 + 140.0

    coords = np.array(instance["coords"], dtype=np.float64)
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    n_nodes = len(coords)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))

    customers = list(range(1, n_customers + 1))
    cust_set = set(customers)

    # --- Fast vectorized route simulation ---
    # Returns (feasible, total_distance). Early exit on first infeasibility.
    def simulate(route):
        if not route:
            return True, 0.0
        r = list(route)
        n = len(r)
        # prev starts at depot (0); next = r
        prev = np.zeros(n + 1, dtype=np.int64)
        prev[0] = 0
        prev[1:] = r
        # dist from prev to next leg
        d_leg = dist[prev[:-1], prev[1:]]          # length n
        cum_t = np.cumsum(d_leg / speed)
        t = cum_t
        bat = battery - np.cumsum(d_leg)
        # time-window: at customer i, t = max(arrive, ready); must be <= due
        t = np.maximum(t, ready[prev[1:]])
        if (t > due[prev[1:]] + 1e-9).any():
            return False, 0.0
        t = t + service[prev[1:]]
        bat = bat - service[prev[1:]] * 0.0  # service doesn't drain battery here (per spec, distance drains)
        # return leg
        d_back = dist[r[-1], 0]
        bat_end = bat[-1] - d_back
        if bat_end < -1e-9:
            return False, 0.0
        t_end = t[-1] + d_back / speed
        if t_end > horizon + 1e-9:
            return False, 0.0
        # battery must never go below 0 along the path (check cumsum already did via bat)
        if (bat < -1e-9).any():
            return False, 0.0
        total_dist = float(d_leg.sum() + d_back)
        return True, total_dist

    def route_distance(route):
        if not route:
            return 0.0
        r = list(route)
        d = dist[0, r[0]]
        if len(r) > 1:
            d += dist[r[:-1], r[1:]].sum()
        d += dist[r[-1], 0]
        return float(d)

    def total_distance(routes):
        return float(sum(route_distance(r) for r in routes if r))

    def validate(routes):
        seen = set()
        for r in routes:
            if not r:
                continue
            ok, _ = simulate(r)
            if not ok:
                return False
            for v in r:
                if v in seen:
                    return False
                seen.add(v)
        return seen == cust_set

    # --- Construction: K-means spatial clustering ---
    def kmeans_construct(k, seed=0):
        rng = np.random.RandomState(seed)
        X = coords[1:n_customers + 1]  # customer coords
        # init: random customers
        idx = rng.choice(n_customers, size=min(k, n_customers), replace=False)
        centers = X[idx].copy()
        if len(centers) < k:
            pad = rng.choice(n_customers, size=k - len(centers), replace=True)
            centers = np.vstack([centers, X[pad]])
        labels = np.zeros(n_customers, dtype=np.int64)
        for _ in range(30):
            d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = d2.argmin(axis=1)
            if np.array_equal(new_labels, labels) and _ > 0:
                break
            labels = new_labels
            for c in range(k):
                mask = labels == c
                if mask.any():
                    centers[c] = X[mask].mean(axis=0)
                else:
                    # reassign an empty cluster to a random customer
                    centers[c] = X[rng.randint(n_customers)]
        routes = []
        for c in range(k):
            mask = labels == c
            if mask.any():
                routes.append(list((np.where(mask)[0] + 1)))  # +1 -> customer ids
        # ensure every customer covered (in case of empty/missing)
        covered = set()
        for r in routes:
            covered.update(r)
        missing = [v for v in customers if v not in covered]
        if missing:
            routes.append(missing)
        return routes

    # --- Construction: Savings ---
    def savings_construct():
        routes = [[v] for v in customers]
        route_of = {v: i for i, v in enumerate(customers)}
        savings = []
        for i in range(n_customers):
            for j in range(i + 1, n_customers):
                vi, vj = i + 1, j + 1
                s = dist[0, vi] + dist[0, vj] - dist[vi, vj]
                if s > 0:
                    savings.append((s, vi, vj))
        savings.sort(reverse=True)
        for s, vi, vj in savings:
            if time.time() > DEADLINE:
                break
            ri = route_of.get(vi)
            rj = route_of.get(vj)
            if ri is None or rj is None or ri == rj:
                continue
            route_i = routes[ri]
            route_j = routes[rj]
            options = []
            if route_i[-1] == vi and route_j[0] == vj:
                options.append(route_i + route_j)
            if route_j[-1] == vj and route_i[0] == vi:
                options.append(route_j + route_i)
            if route_i[0] == vi and route_j[0] == vj:
                options.append(route_i[::-1] + route_j)
            if route_i[-1] == vi and route_j[-1] == vj:
                options.append(route_i + route_j[::-1])
            if route_i[0] == vi and route_j[-1] == vj:
                options.append(route_i[::-1] + route_j[::-1])
            if route_j[0] == vj and route_i[-1] == vi:
                options.append(route_j + route_i[::-1])
            for candidate in options:
                if sum(demands[x] for x in candidate) > capacity:
                    continue
                f, _ = simulate(candidate)
                if f:
                    routes[ri] = candidate
                    routes[rj] = []
                    non_empty = [r for r in routes if r]
                    route_of = {}
                    for i, r in enumerate(non_empty):
                        for v in r:
                            route_of[v] = i
                    routes = non_empty
                    break
        return [r for r in routes if r]

    # --- Local search: 2-opt ---
    def two_opt_route(route, max_iters=50):
        if len(route) < 2:
            return route
        route = list(route)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            n = len(route)
            for a in range(n - 1):
                if time.time() > DEADLINE:
                    return route
                for b in range(a + 1, n):
                    old_b = (dist[0, route[a]] if a == 0 else dist[route[a-1], route[a]]) + \
                            (dist[route[b], 0] if b == n-1 else dist[route[b], route[b+1]])
                    new_b = (dist[0, route[b]] if a == 0 else dist[route[a-1], route[b]]) + \
                            (dist[route[a], 0] if b == n-1 else dist[route[a], route[b+1]])
                    if new_b < old_b - 1e-6:
                        new_route = route[:a] + route[a:b+1][::-1] + route[b+1:]
                        f, _ = simulate(new_route)
                        if f:
                            route = new_route
                            improved = True
                            break
                if improved:
                    break
        return route

    # --- Local search: Or-opt (move block of 1..3 consecutive) ---
    def or_opt_route(route, max_iters=40):
        if len(route) < 2:
            return route
        route = list(route)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            n = len(route)
            for block in (1, 2, 3):
                if n < block + 1:
                    continue
                for start in range(n - block + 1):
                    if time.time() > DEADLINE:
                        return route
                    seg = route[start:start+block]
                    rest = route[:start] + route[start+block:]
                    if not rest:
                        continue
                    # try inserting seg into every position of rest
                    for pos in range(len(rest) + 1):
                        cand = rest[:pos] + seg + rest[pos:]
                        # skip trivial no-op
                        if cand == route:
                            continue
                        f, _ = simulate(cand)
                        if f:
                            cd = route_distance(cand)
                            od = route_distance(route)
                            if cd < od - 1e-6:
                                route = cand
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        return route

    # --- Local search: relocate (single customer) ---
    def relocate_search(routes, max_passes=4):
        routes = [list(r) for r in routes]
        for _ in range(max_passes):
            if time.time() > DEADLINE:
                break
            improved = False
            cur = total_distance(routes)
            for i in range(len(routes)):
                if time.time() > DEADLINE:
                    break
                for j in range(len(routes[i])):
                    if time.time() > DEADLINE:
                        break
                    v = routes[i][j]
                    ri = routes[i][:j] + routes[i][j+1:]
                    if ri:
                        f, _ = simulate(ri)
                        if not f:
                            continue
                    for k in range(len(routes)):
                        if time.time() > DEADLINE:
                            break
                        rk = routes[k]
                        for pos in range(len(rk) + 1):
                            cand = rk[:pos] + [v] + rk[pos:]
                            f, _ = simulate(cand)
                            if not f:
                                continue
                            # build new routes
                            new_routes = [list(routes[m]) for m in range(len(routes))]
                            if k == i:
                                new_routes[i] = ri[:pos] + [v] + ri[pos:] if ri else [v]
                            else:
                                new_routes[i] = ri
                                new_routes[k] = cand
                            if not new_routes[i]:
                                new_routes = [r for idx, r in enumerate(new_routes) if r or idx != i]
                                new_routes = [r for r in new_routes if r]
                            nd = total_distance(new_routes)
                            if nd < cur - 1e-6:
                                routes = new_routes
                                cur = nd
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

    # --- Local search: swap (exchange two customers between routes) ---
    def swap_search(routes, max_passes=2):
        for _ in range(max_passes):
            if time.time() > DEADLINE:
                break
            improved = False
            cur = total_distance(routes)
            for i in range(len(routes)):
                if time.time() > DEADLINE:
                    break
                for j in range(i + 1, len(routes)):
                    if time.time() > DEADLINE:
                        break
                    for a in range(len(routes[i])):
                        if time.time() > DEADLINE:
                            break
                        for b in range(len(routes[j])):
                            vi, vj = routes[i][a], routes[j][b]
                            new_routes = [list(routes[m]) for m in range(len(routes))]
                            new_routes[i][a] = vj
                            new_routes[j][b] = vi
                            ok = True
                            for r in new_routes:
                                f, _ = simulate(r)
                                if not f:
                                    ok = False
                                    break
                            if ok:
                                nd = total_distance(new_routes)
                                if nd < cur - 1e-6:
                                    routes = new_routes
                                    cur = nd
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

    # --- Build initial solutions and pick the best ---
    candidates = []

    # K-means with several k and seeds
    for k in (4, 5, 6, 7, 8):
        for seed in (0, 1):
            if time.time() > DEADLINE:
                break
            candidates.append(kmeans_construct(k, seed))

    # Savings
    if time.time() < DEADLINE:
        candidates.append(savings_construct())

    # Greedy insertion by due date
    def greedy_insertion(ordering):
        routes = []
        remaining = list(ordering)
        while remaining:
            v = remaining.pop(0)
            best_ri, best_pos, best_cost = -1, -1, float('inf')
            for ri, route in enumerate(routes):
                for pos in range(len(route) + 1):
                    new_route = route[:pos] + [v] + route[pos:]
                    if sum(demands[x] for x in new_route) > capacity:
                        continue
                    f, _ = simulate(new_route)
                    if not f:
                        continue
                    if pos == 0:
                        old_d = dist[0, route[0]]
                        new_d = dist[0, v] + dist[v, route[0]]
                    elif pos == len(route):
                        old_d = dist[route[-1], 0]
                        new_d = dist[route[-1], v] + dist[v, 0]
                    else:
                        old_d = dist[route[pos-1], route[pos]]
                        new_d = dist[route[pos-1], v] + dist[v, route[pos]]
                    cost = new_d - old_d
                    if cost < best_cost:
                        best_cost = cost
                        best_ri, best_pos = ri, pos
            if best_ri == -1:
                routes.append([v])
            else:
                routes[best_ri] = routes[best_ri][:best_pos] + [v] + routes[best_ri][best_pos:]
        return routes

    orderings = [
        sorted(customers, key=lambda v: due[v]),
        sorted(customers, key=lambda v: ready[v]),
        sorted(customers, key=lambda v: dist[0, v]),
        sorted(customers, key=lambda v: (due[v], dist[0, v])),
    ]
    for ordering in orderings:
        if time.time() > DEADLINE:
            break
        candidates.append(greedy_insertion(ordering))

    # Keep only valid, pick best
    valid = []
    for c in candidates:
        if not c:
            continue
        if validate(c):
            valid.append(c)
    if not valid:
        # Fallback: one customer per route (always feasible)
        valid = [[v] for v in customers]

    best = min(valid, key=lambda r: total_distance(r))
    best_dist = total_distance(best)

    # --- Local search refinement loop ---
    # Alternate 2-opt + Or-opt per route, then relocate, then swap; repeat while time allows.
    rounds = 0
    while time.time() < DEADLINE - 5.0 and rounds < 6:
        rounds += 1
        # Intra-route: 2-opt then Or-opt
        for i in range(len(best)):
            if time.time() > DEADLINE:
                break
            best[i] = two_opt_route(best[i])
            best[i] = or_opt_route(best[i])
        # Inter-route: relocate then swap
        if time.time() < DEADLINE:
            best = relocate_search(best, max_passes=3)
        if time.time() < DEADLINE:
            best = swap_search(best, max_passes=1)
        # ensure validity; if broken, revert to previous valid
        if not validate(best):
            # fallback: rebuild from last valid by keeping one-customer-per-route? Instead, re-validate each route and drop invalid
            # Simpler: fall back to a valid construction
            fb = kmeans_construct(5, 0)
            if validate(fb):
                best = fb
            else:
                best = [[v] for v in customers]
            break

    # Final polish: one more 2-opt + Or-opt pass
    if time.time() < DEADLINE - 2.0:
        for i in range(len(best)):
            if time.time() > DEADLINE:
                break
            best[i] = two_opt_route(best[i], max_iters=30)
            best[i] = or_opt_route(best[i], max_iters=25)

    # Final validity check; if invalid, return safe fallback
    if not validate(best):
        best = [[v] for v in customers]

    best = [r for r in best if r]
    return best
# EVOLVE-BLOCK-END
