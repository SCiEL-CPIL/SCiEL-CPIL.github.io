"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def plan(instance):
    start = time.time()
    deadline = start + 140.0  # safety margin vs 150s wall-clock

    coords = np.asarray(instance["coords"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    ready = np.asarray(instance["ready"], dtype=float)
    due = np.asarray(instance["due"], dtype=float)
    service = np.asarray(instance["service"], dtype=float)
    n_customers = instance["n_customers"]
    stations = list(instance["stations"])
    capacity = float(instance["capacity"])
    battery_cap = float(instance["battery"])
    speed = float(instance["speed"])
    recharge_rate = float(instance["recharge_rate"])
    horizon = float(instance["horizon"])

    n = coords.shape[0]
    # Precompute squared distances and Euclidean distances.
    diff = coords[:, None, :] - coords[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    dist = np.sqrt(d2)

    customers = list(range(1, n_customers + 1))

    # ---------- Feasibility simulation (customers only) ----------
    def simulate(route):
        """Return (ok, total_dist, end_time). Battery/time windows/capacity/horizon."""
        load = 0.0
        for v in route:
            load += demands[v]
        if load > capacity + 1e-9:
            return False, 0.0, 0.0
        t = 0.0
        battery = battery_cap
        total = 0.0
        prev = 0
        for v in route:
            d = dist[prev, v]
            if d > battery + 1e-9:
                return False, 0.0, 0.0
            battery -= d
            t += d / speed
            total += d
            if t < ready[v]:
                t = ready[v]
            if t > due[v] + 1e-9:
                return False, 0.0, 0.0
            t += service[v]
            prev = v
        d = dist[prev, 0]
        if d > battery + 1e-9:
            return False, 0.0, 0.0
        t += d / speed
        if t > horizon + 1e-9:
            return False, 0.0, 0.0
        return True, total, t

    # ---------- Greedy construction: build K routes ----------
    def construct(k_routes, seed):
        rng = random.Random(seed)
        remaining = list(customers)
        rng.shuffle(remaining)
        # order remaining by earliest due first for stable insertion
        remaining.sort(key=lambda v: due[v])
        routes = [[] for _ in range(k_routes)]
        for v in remaining:
            # try to insert into the route that yields least extra distance & feasible
            best = None
            for r in range(k_routes):
                route = routes[r]
                L = len(route)
                # candidate positions
                positions = list(range(L + 1))
                for pos in positions:
                    # quick capacity check
                    new_load = sum(demands[x] for x in route) + demands[v]
                    if new_load > capacity + 1e-9:
                        continue
                    new_route = route[:pos] + [v] + route[pos:]
                    ok, tot, _ = simulate(new_route)
                    if ok:
                        if best is None or tot < best[1]:
                            best = (r, tot, new_route)
            if best is not None:
                routes[best[0]] = best[2]
            else:
                # cannot fit anywhere: start a new route (should not happen with enough routes)
                routes.append([v])
        # drop empty routes
        routes = [r for r in routes if r]
        return routes

    # ---------- 2-opt within a route ----------
    def route_dist(route):
        if not route:
            return 0.0
        t = dist[0, route[0]]
        for i in range(len(route) - 1):
            t += dist[route[i], route[i + 1]]
        t += dist[route[-1], 0]
        return t

    def two_opt(route, max_iter=2000):
        if len(route) < 3:
            return route
        best = list(route)
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            L = len(best)
            for i in range(L - 1):
                for j in range(i + 1, L):
                    a, b = best[i - 1] if i > 0 else 0, best[i]
                    c, d = best[j], best[j + 1] if j + 1 < L else 0
                    # reversal of best[i..j]
                    old = dist[a, b] + dist[c, d]
                    new = dist[a, c] + dist[b, d]
                    if new < old - 1e-9:
                        cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                        ok, tot, _ = simulate(cand)
                        if ok:
                            best = cand
                            improved = True
                            break
                if improved:
                    break
        return best

    # ---------- Or-opt: move a segment of length 1-3 ----------
    def or_opt(route, max_iter=1000):
        if len(route) < 3:
            return route
        best = list(route)
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            L = len(best)
            for seg_len in (1, 2, 3):
                for i in range(L - seg_len + 1):
                    seg = best[i:i + seg_len]
                    rest = best[:i] + best[i + seg_len:]
                    if not rest:
                        continue
                    # try inserting seg at each position in rest
                    for pos in range(len(rest) + 1):
                        cand = rest[:pos] + seg + rest[pos:]
                        if cand == best:
                            continue
                        ok, tot, _ = simulate(cand)
                        if ok:
                            old = route_dist(best)
                            new = route_dist(cand)
                            if new < old - 1e-9:
                                best = cand
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        return best

    # ---------- Evaluate ----------
    def total_distance(routes):
        return sum(route_dist(r) for r in routes)

    # Choose initial number of routes based on total demand.
    total_demand = sum(demands[v] for v in customers)
    min_routes = max(1, int(math.ceil(total_demand / capacity)))
    base_k = min_routes + 1

    best_routes = None
    best_score = float('inf')

    # Try a few route counts, keep best.
    k_candidates = sorted(set([base_k, base_k + 1, base_k + 2, min_routes]))
    for k in k_candidates:
        if time.time() > deadline - 5.0:
            break
        for seed in range(6):
            if time.time() > deadline - 5.0:
                break
            routes = construct(k, seed)
            # local search
            for r in range(len(routes)):
                if time.time() > deadline - 5.0:
                    break
                routes[r] = two_opt(routes[r])
                routes[r] = or_opt(routes[r])
            score = total_distance(routes)
            if score < best_score:
                best_score = score
                best_routes = [list(r) for r in routes]

    # Fallback: ensure every customer served; if construction failed, one per route.
    if best_routes is None:
        best_routes = [[v] for v in customers]
    else:
        served = set()
        for r in best_routes:
            for v in r:
                served.add(v)
        missing = [v for v in customers if v not in served]
        for v in missing:
            best_routes.append([v])

    return best_routes
# EVOLVE-BLOCK-END
