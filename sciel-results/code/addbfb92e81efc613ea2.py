# EVOLVE-BLOCK-START
import math
import time
import numpy as np


def plan(instance):
    t0 = time.perf_counter()
    DEADLINE = 135.0  # safety margin under the 150s kill

    n_cust = instance["n_customers"]
    coords = np.asarray(instance["coords"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    ready = np.asarray(instance["ready"], dtype=float)
    due = np.asarray(instance["due"], dtype=float)
    service = np.asarray(instance["service"], dtype=float)
    stations = list(instance["stations"])
    capacity = float(instance["capacity"])
    battery = float(instance["battery"])
    speed = float(instance["speed"])
    rr = float(instance["recharge_rate"])
    horizon = float(instance["horizon"])

    c = coords
    n = c.shape[0]
    dx = c[:, [0]] - c[[0], :]
    dy = c[:, [1]] - c[[1], :]
    D = np.sqrt(dx * dx + dy * dy)

    customers = list(range(1, n_cust + 1))
    st_set = set(stations)

    def simulate(route):
        t = 0.0
        b = battery
        d = 0.0
        load = 0.0
        prev = 0
        for v in route:
            d += D[prev, v]
            t += D[prev, v] / speed
            b -= D[prev, v]
            if b < -1e-9:
                return False, d
            if v in st_set:
                if b < 0:
                    b = 0.0
                t += (battery - b) * rr
                b = battery
            else:
                if t < ready[v]:
                    t = ready[v]
                if t > due[v] + 1e-9:
                    return False, d
                t += service[v]
                load += demands[v]
                if load > capacity + 1e-9:
                    return False, d
            prev = v
        d += D[prev, 0]
        t += D[prev, 0] / speed
        b -= D[prev, 0]
        if b < -1e-9:
            return False, d
        if t > horizon + 1e-9:
            return False, d
        return True, d

    def build_routes():
        routes = []
        for seed in customers:
            route = [seed]
            remaining = [x for x in customers if x != seed]
            while remaining:
                last = route[-1]
                best = -1
                best_d = float("inf")
                for x in remaining:
                    dd = D[last, x]
                    if dd < best_d and simulate(route + [x])[0]:
                        best = x
                        best_d = dd
                if best == -1:
                    break
                route.append(best)
                remaining.remove(best)
            routes.append(route)
        return routes

    def total_distance(routes):
        return sum(simulate(r)[1] for r in routes)

    def merge(routes, i, j, order):
        a = routes[i]
        b = routes[j]
        if order == "ij":
            cands = [a + b, a + list(reversed(b))]
        else:
            cands = [b + a, b + list(reversed(a))]
        for cand in cands:
            if simulate(cand)[0]:
                new = [r for idx, r in enumerate(routes) if idx not in (i, j)]
                new.append(cand)
                return new
        if order == "ij":
            parts = [a, b]
        else:
            parts = [b, a]
        for s in stations:
            cand = parts[0] + [s] + parts[1]
            if simulate(cand)[0]:
                new = [r for idx, r in enumerate(routes) if idx not in (i, j)]
                new.append(cand)
                return new
        return None

    def construct():
        routes = build_routes()
        improved = True
        while improved:
            improved = False
            base = total_distance(routes)
            best_gain = 1e-9
            best = None
            nr = len(routes)
            for i in range(nr):
                for j in range(i + 1, nr):
                    for order in ("ij", "ji"):
                        merged = merge(routes, i, j, order)
                        if merged is None:
                            continue
                        nd = total_distance(merged)
                        gain = base - nd
                        if gain > best_gain:
                            best_gain = gain
                            best = merged
            if best is not None:
                routes = best
                improved = True
        return routes

    def two_opt(route):
        m = len(route)
        if m < 2:
            return route
        best = route
        best_d = simulate(route)[1]
        for i in range(m - 1):
            for j in range(i + 1, m):
                cand = route[:i] + list(reversed(route[i:j + 1])) + route[j + 1:]
                ok, d = simulate(cand)
                if ok and d < best_d - 1e-9:
                    best = cand
                    best_d = d
        return best

    def relocate(routes):
        best = routes
        best_d = total_distance(routes)
        nr = len(routes)
        for k in range(nr):
            for pos in range(len(routes[k])):
                c = routes[k][pos]
                for k2 in range(nr):
                    if k2 == k:
                        continue
                    for pos2 in range(len(routes[k2]) + 1):
                        src = list(routes[k])
                        src.pop(pos)
                        if not src:
                            continue
                        if not simulate(src)[0]:
                            continue
                        dst = list(routes[k2])
                        dst.insert(pos2, c)
                        if not simulate(dst)[0]:
                            continue
                        new = list(routes)
                        new[k] = src
                        new[k2] = dst
                        nd = total_distance(new)
                        if nd < best_d - 1e-9:
                            best_d = nd
                            best = new
        return best

    def swap(routes):
        best = routes
        best_d = total_distance(routes)
        nr = len(routes)
        for k in range(nr):
            for k2 in range(k + 1, nr):
                for p in range(len(routes[k])):
                    for q in range(len(routes[k2])):
                        a = routes[k][p]
                        b = routes[k2][q]
                        r1 = list(routes[k])
                        r2 = list(routes[k2])
                        r1[p], r2[q] = b, a
                        if not (simulate(r1)[0] and simulate(r2)[0]):
                            continue
                        new = list(routes)
                        new[k] = r1
                        new[k2] = r2
                        nd = total_distance(new)
                        if nd < best_d - 1e-9:
                            best_d = nd
                            best = new
        return best

    routes = construct()
    best_routes = routes
    best_d = total_distance(routes)

    while time.perf_counter() - t0 < DEADLINE:
        prev = total_distance(routes)
        changed = False

        r2 = relocate(routes)
        if total_distance(r2) < total_distance(routes) - 1e-9:
            routes = r2
            changed = True
        r3 = swap(routes)
        if total_distance(r3) < total_distance(routes) - 1e-9:
            routes = r3
            changed = True
        for idx in range(len(routes)):
            nr = two_opt(routes[idx])
            if simulate(nr)[1] < simulate(routes[idx])[1] - 1e-9:
                routes[idx] = nr
                changed = True

        cur = total_distance(routes)
        if cur < best_d:
            best_d = cur
            best_routes = routes
        if not changed and cur >= prev - 1e-9:
            break

    return best_routes
# EVOLVE-BLOCK-END
