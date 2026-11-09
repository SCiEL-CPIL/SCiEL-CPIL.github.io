"""EV routing with time windows + battery; constructive + local search."""
# EVOLVE-BLOCK-START
import math
import random
import time

import numpy as np


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _sim(route, C, demand, ready, due, service, st, B, RR, H, speed=1.0, eps=1e-9):
    """Simulate one route (depot implicit both ends). Returns (ok, dist, end_time)."""
    t = 0.0
    bat = B
    dist = 0.0
    cur = 0
    for v in route:
        d = _dist(C[cur], C[v])
        t += d / speed
        bat -= d
        dist += d
        if bat < -eps:
            return False, dist, t
        if v in st:
            t += max(0.0, (B - bat)) * RR
            bat = B
        else:
            if t < ready[v]:
                t = ready[v]
            if t > due[v] + eps:
                return False, dist, t
            t += service[v]
        cur = v
    d = _dist(C[cur], C[0])
    t += d / speed
    bat -= d
    dist += d
    if bat < -eps or t > H + eps:
        return False, dist, t
    return True, dist, t


def _route_cost(route, C):
    d = 0.0
    cur = 0
    for v in route:
        d += _dist(C[cur], C[v])
        cur = v
    d += _dist(C[cur], C[0])
    return d


def _check_all(routes, C, demand, ready, due, service, st, B, RR, H, speed=1.0):
    served = [0] * (len(C))
    for r in routes:
        for v in r:
            served[v] += 1
        if not _sim(r, C, demand, ready, due, service, st, B, RR, H, speed)[0]:
            return False
    for i in range(1, len(C)):
        if served[i] != 1:
            return False
    return True


def _try_insert(r, pos, v, C, demand, ready, due, service, st, B, RR, H, speed=1.0):
    """Return (ok, new_route) if inserting v at pos keeps route feasible."""
    nr = r[:pos] + [v] + r[pos:]
    return _sim(nr, C, demand, ready, due, service, st, B, RR, H, speed)[0], nr


def _construct(C, demand, ready, due, service, st, B, RR, H, speed=1.0, seed=0):
    rng = random.Random(seed)
    n = len(C)
    cust = list(range(1, n))
    # sort by due (tight first) with small jitter
    cust.sort(key=lambda v: due[v] + rng.uniform(0, 20))
    routes = []
    for v in cust:
        best = None  # (delta, route_idx, pos, new_route)
        for ri, r in enumerate(routes):
            for pos in range(len(r) + 1):
                ok, nr = _try_insert(r, pos, v, C, demand, ready, due, service, st, B, RR, H, speed)
                if ok:
                    delta = _route_cost(nr, C) - _route_cost(r, C)
                    if best is None or delta < best[0]:
                        best = (delta, ri, pos, nr)
        if best is not None:
            routes[best[1]] = best[3]
        else:
            ok, _ = _sim([v], C, demand, ready, due, service, st, B, RR, H, speed)
            if ok:
                routes.append([v])
            else:
                # infeasible single; try via nearest station first
                placed = False
                for s in st:
                    ok, _ = _sim([s, v], C, demand, ready, due, service, st, B, RR, H, speed)
                    if ok:
                        routes.append([s, v])
                        placed = True
                        break
                if not placed:
                    # give up on this customer (will be caught by validator)
                    pass
    return routes


def _or_opt(r, C, demand, ready, due, service, st, B, RR, H, speed=1.0):
    """Remove 1-2 consecutive nodes and reinsert best within same route. Returns (improved, new_r)."""
    n = len(r)
    if n < 3:
        return False, r
    for L in (1, 2):
        for i in range(0, n - L + 1):
            seg = r[i:i + L]
            rest = r[:i] + r[i + L:]
            base = _route_cost(r, C)
            best = (base, r)
            for pos in range(len(rest) + 1):
                cand = rest[:pos] + seg + rest[pos:]
                if _sim(cand, C, demand, ready, due, service, st, B, RR, H, speed)[0]:
                    c = _route_cost(cand, C)
                    if c < best[0] - 1e-9:
                        best = (c, cand)
            if best[0] < base - 1e-9:
                return True, best[1]
    return False, r


def _two_opt(r, C, demand, ready, due, service, st, B, RR, H, speed=1.0):
    n = len(r)
    if n < 3:
        return False, r
    base = _route_cost(r, C)
    best = (base, r)
    for i in range(1, n - 1):
        for j in range(i + 1, n):
            cand = r[:i] + r[i:j + 1][::-1] + r[j + 1:]
            if _sim(cand, C, demand, ready, due, service, st, B, RR, H, speed)[0]:
                c = _route_cost(cand, C)
                if c < best[0] - 1e-9:
                    best = (c, cand)
    if best[0] < base - 1e-9:
        return True, best[1]
    return False, r


def _improve(routes, C, demand, ready, due, service, st, B, RR, H, speed=1.0, deadline=None):
    improved = True
    while improved:
        improved = False
        if deadline is not None and time.time() > deadline:
            break
        for ri in range(len(routes)):
            r = routes[ri]
            if len(r) < 3:
                continue
            imp, nr = _or_opt(r, C, demand, ready, due, service, st, B, RR, H, speed)
            if imp:
                routes[ri] = nr
                improved = True
                continue
            imp, nr = _two_opt(r, C, demand, ready, due, service, st, B, RR, H, speed)
            if imp:
                routes[ri] = nr
                improved = True
        # inter-route: try moving a node from route a to route b
        if deadline is not None and time.time() > deadline:
            break
        for ai in range(len(routes)):
            ra = routes[ai]
            for i in range(len(ra)):
                v = ra[i]
                if v in st:
                    continue
                rest_a = ra[:i] + ra[i + 1:]
                if not _sim(rest_a, C, demand, ready, due, service, st, B, RR, H, speed)[0]:
                    continue
                base = _route_cost(ra, C)
                for bi in range(len(routes)):
                    if bi == ai:
                        continue
                    rb = routes[bi]
                    for pos in range(len(rb) + 1):
                        ok, nr_b = _try_insert(rb, pos, v, C, demand, ready, due, service, st, B, RR, H, speed)
                        if ok:
                            delta = (_route_cost(rest_a, C) + _route_cost(nr_b, C)) - (base + _route_cost(rb, C))
                            if delta < -1e-9:
                                routes[ai] = rest_a
                                routes[bi] = nr_b
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    # drop empty routes
    routes = [r for r in routes if len(r) > 0]
    return routes


def plan(instance):
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin under 150s kill

    C = instance["coords"]
    demand = instance["demand"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n = instance["n_customers"]
    st = set(instance["stations"])
    B = instance["battery"]
    RR = instance["recharge_rate"]
    H = instance["horizon"]
    speed = instance.get("speed", 1.0)

    best_routes = None
    best_score = None

    for seed in range(0, 400):
        if time.time() > deadline:
            break
        routes = _construct(C, demand, ready, due, service, st, B, RR, H, speed, seed)
        if not _check_all(routes, C, demand, ready, due, service, st, B, RR, H, speed):
            continue
        routes = _improve(routes, C, demand, ready, due, service, st, B, RR, H, speed, deadline)
        if not _check_all(routes, C, demand, ready, due, service, st, B, RR, H, speed):
            continue
        score = sum(_route_cost(r, C) for r in routes)
        if best_score is None or score < best_score:
            best_score = score
            best_routes = [list(r) for r in routes]

    if best_routes is None:
        # fallback: one customer per route, earliest due first (always feasible w/ full battery)
        cust = list(range(1, n + 1))
        cust.sort(key=lambda v: due[v])
        best_routes = [[v] for v in cust]

    return best_routes
# EVOLVE-BLOCK-END
