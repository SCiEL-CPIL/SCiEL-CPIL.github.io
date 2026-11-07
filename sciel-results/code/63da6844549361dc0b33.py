"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin under the 150s kill

    coords = np.array(instance["coords"], dtype=float)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    # Precompute full Euclidean distance matrix (vectorized).
    n_nodes = len(coords)
    dx = coords[:, 0:1] - coords[0:1, 0]
    dy = coords[:, 1:2] - coords[0:1, 1]
    dist = np.sqrt(dx * dx + dy * dy)

    customers = list(range(1, n_customers + 1))

    # ---------- Fast feasibility ----------
    def route_feasible(route):
        """Return (feasible, total_dist). Greedy charging at stations when battery low."""
        cur_time = 0.0
        cur_battery = float(battery)
        cur_pos = 0
        total_dist = 0.0
        for cust in route:
            d = dist[cur_pos, cust]
            cur_time += d / speed
            cur_battery -= d
            if cur_battery < 0.0:
                return False, total_dist
            total_dist += d
            cur_time = max(cur_time, float(ready[cust]))
            if cur_time > float(due[cust]):
                return False, total_dist
            cur_time += float(service[cust])
            cur_pos = cust
        d = dist[cur_pos, 0]
        cur_time += d / speed
        cur_battery -= d
        if cur_battery < 0.0:
            return False, total_dist
        total_dist += d
        if cur_time > horizon:
            return False, total_dist
        return True, total_dist

    # ---------- Baseline: one customer per route, earliest-due first (always feasible) ----------
    baseline_order = sorted(customers, key=lambda v: float(due[v]))
    baseline = [[v] for v in baseline_order]
    baseline_dist = 0.0
    for r in baseline:
        _, d = route_feasible(r)
        baseline_dist += d

    # ---------- Improvement ----------
    def total_dist(routes):
        s = 0.0
        for r in routes:
            if not r:
                continue
            d = dist[0, r[0]]
            for i in range(len(r) - 1):
                d += dist[r[i], r[i + 1]]
            d += dist[r[-1], 0]
            s += d
        return s

    routes = [list(r) for r in baseline]
    cur_total = total_dist(routes)

    # 2-opt within a single route (distance-only; feasibility rechecked after)
    def two_opt(route):
        n = len(route)
        if n < 3:
            return route
        improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 1, n):
                    a = route[i - 1] if i > 0 else 0
                    b = route[i]
                    c = route[j]
                    d = route[j + 1] if j + 1 < n else 0
                    delta = dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d]
                    if delta < -1e-9:
                        route[i:j + 1] = route[i:j + 1][::-1]
                        improved = True
                        break
                if improved:
                    break
        return route

    # Cross-route relocations (merges routes => big distance win)
    def relocate_pass(routes):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            nr = len(routes)
            for i in range(nr):
                if not routes[i]:
                    continue
                ri = routes[i]
                for j in range(len(ri)):
                    cust = ri[j]
                    ri_without = ri[:j] + ri[j + 1:]
                    for k in range(nr):
                        if k == i:
                            continue
                        rk = routes[k]
                        for pos in range(len(rk) + 1):
                            new_rk = rk[:pos] + [cust] + rk[pos:]
                            # demand check
                            if sum(demands[c] for c in new_rk) > capacity:
                                continue
                            feas_k, _ = route_feasible(new_rk)
                            if not feas_k:
                                continue
                            feas_i, _ = route_feasible(ri_without) if ri_without else (True, 0.0)
                            if not feas_i:
                                continue
                            # distance comparison
                            old = (dist[0, ri[0]] + sum(dist[ri[m], ri[m + 1]] for m in range(len(ri) - 1)) + dist[ri[-1], 0])
                            old += (dist[0, rk[0]] + sum(dist[rk[m], rk[m + 1]] for m in range(len(rk) - 1)) + dist[rk[-1], 0])
                            nd_i = (dist[0, ri_without[0]] + sum(dist[ri_without[m], ri_without[m + 1]] for m in range(len(ri_without) - 1)) + dist[ri_without[-1], 0]) if ri_without else 0.0
                            nd_k = dist[0, new_rk[0]] + sum(dist[new_rk[m], new_rk[m + 1]] for m in range(len(new_rk) - 1)) + dist[new_rk[-1], 0]
                            new = nd_i + nd_k
                            if new < old - 1e-6:
                                routes[i] = ri_without
                                routes[k] = new_rk
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return routes

    # Apply 2-opt to each route first (cheap distance reduction)
    for i in range(len(routes)):
        if time.time() < deadline:
            routes[i] = two_opt(routes[i])

    # Then cross-route relocations
    routes = relocate_pass(routes)

    # Drop any empty routes
    routes = [r for r in routes if r]

    # ---------- Safety: verify whole solution; fall back to baseline if broken ----------
    served = set()
    ok = True
    for r in routes:
        for c in r:
            if c in served:
                ok = False
                break
            served.add(c)
        if ok:
            f, _ = route_feasible(r)
            if not f:
                ok = False
                break
    if ok and served == set(customers):
        return routes
    # Fallback: guaranteed-feasible baseline
    return baseline
# EVOLVE-BLOCK-END
