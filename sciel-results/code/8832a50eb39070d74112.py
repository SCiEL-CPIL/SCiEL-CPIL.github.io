"""EV routing with time windows + battery; Clarke-Wright + charging insertion + local search."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np


def _build(instance):
    coords = np.asarray(instance["coords"], dtype=float)
    n = len(coords)
    diff = coords[:, None, :] - coords[None, :, :]
    D = np.sqrt((diff ** 2).sum(axis=2))
    demands = np.asarray(instance["demands"], dtype=float)
    ready = np.asarray(instance["ready"], dtype=float)
    due = np.asarray(instance["due"], dtype=float)
    service = np.asarray(instance["service"], dtype=float)
    n_cust = instance["n_customers"]
    stations = list(instance["stations"])
    cap = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    rr = instance["recharge_rate"]
    horizon = instance["horizon"]
    return (D, demands, ready, due, service, n, n_cust, stations,
            cap, battery, speed, rr, horizon)


def _sim(seq, D, demands, ready, due, service, battery, speed, rr, horizon):
    t = 0.0
    bat = float(battery)
    load = 0.0
    dist = 0.0
    prev = 0
    for v in seq:
        d = float(D[prev, v])
        t += d / speed
        bat -= d
        dist += d
        if bat < -1e-9:
            return None
        if service[v] > 0:
            if t < ready[v]:
                t = float(ready[v])
            if t > due[v] + 1e-9:
                return None
            t += float(service[v])
            load += float(demands[v])
            if load > 100.0 + 1e-9:
                return None
        else:
            energy = -bat
            t += energy * rr
            bat = float(battery)
        prev = v
    d = float(D[prev, 0])
    t += d / speed
    bat -= d
    dist += d
    if bat < -1e-9 or t > horizon + 1e-9:
        return None
    return dist


def _dist_seq(seq, D):
    total = 0.0
    prev = 0
    for v in seq:
        total += float(D[prev, v])
        prev = v
    total += float(D[prev, 0])
    return total


def _feasible(seq, D, demands, ready, due, service, battery, speed, rr, horizon):
    return _sim(seq, D, demands, ready, due, service, battery, speed, rr, horizon) is not None


def _insert_station(seq, si, pos, D, demands, ready, due, service, battery, speed, rr, horizon):
    new = seq[:pos] + [si] + seq[pos:]
    return _sim(new, D, demands, ready, due, service, battery, speed, rr, horizon)


def _fix_stations(seq, D, demands, ready, due, service, battery, speed, rr, horizon, stations):
    """Insert charging stations until route is feasible (or give up)."""
    seq = list(seq)
    for _ in range(20):
        if _feasible(seq, D, demands, ready, due, service, battery, speed, rr, horizon):
            return seq
        # find first battery violation
        t = 0.0
        bat = float(battery)
        prev = 0
        viol = None
        for idx, v in enumerate(seq):
            d = float(D[prev, v])
            t += d / speed
            bat -= d
            if bat < -1e-9:
                viol = (idx, t, bat)
                break
            if service[v] > 0:
                if t < ready[v]:
                    t = float(ready[v])
                if t > due[v] + 1e-9:
                    break
                t += float(service[v])
            else:
                energy = -bat
                t += energy * rr
                bat = float(battery)
            prev = v
        if viol is None:
            # time or due violation; try inserting a station right before the
            # customer that's most delayed, or at midpoint
            # find the due-violating customer
            t = 0.0
            bat = float(battery)
            prev = 0
            due_viol = None
            for idx, v in enumerate(seq):
                d = float(D[prev, v])
                t += d / speed
                bat -= d
                if service[v] > 0:
                    if t < ready[v]:
                        t = float(ready[v])
                    if t > due[v] + 1e-9:
                        due_viol = idx
                        break
                    t += float(service[v])
                else:
                    energy = -bat
                    t += energy * rr
                    bat = float(battery)
                prev = v
            if due_viol is None:
                # horizon violation: insert station near the end to save? can't save time.
                # Give up.
                return None
            pos = due_viol
        else:
            idx = viol[0]
            pos = idx
        # choose best station and position around pos
        best = None
        for si in stations:
            for p in range(pos, min(pos + 2, len(seq) + 1)):
                r = _insert_station(seq, si, p, D, demands, ready, due, service,
                                    battery, speed, rr, horizon)
                if r is not None:
                    if best is None or r < best[0]:
                        best = (r, list(seq[:p] + [si] + seq[p:]))
        if best is None:
            return None
        seq = best[1]
    if _feasible(seq, D, demands, ready, due, service, battery, speed, rr, horizon):
        return seq
    return None


def _two_opt(seq, D, demands, ready, due, service, battery, speed, rr, horizon):
    n = len(seq)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                new = seq[:i] + list(reversed(seq[i:j + 1])) + seq[j + 1:]
                r = _sim(new, D, demands, ready, due, service, battery, speed, rr, horizon)
                if r is not None and r < _dist_seq(seq, D) - 1e-9:
                    seq = new
                    improved = True
                    break
            if improved:
                break
    return seq


def _or_opt(seq, D, demands, ready, due, service, battery, speed, rr, horizon):
    n = len(seq)
    improved = True
    while improved:
        improved = False
        for L in (1, 2):
            for i in range(n - L + 1):
                seg = seq[i:i + L]
                rest = seq[:i] + seq[i + L:]
                for p in range(len(rest) + 1):
                    new = rest[:p] + seg + rest[p:]
                    r = _sim(new, D, demands, ready, due, service, battery, speed, rr, horizon)
                    if r is not None and r < _dist_seq(seq, D) - 1e-9:
                        seq = new
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return seq


def _clarke_wright(customers, D, demands, ready, due, service, battery, speed, rr, horizon, stations, cap):
    # start with singletons
    routes = [[c] for c in customers]
    # savings
    n = len(customers)
    idx = {c: i for i, c in enumerate(customers)}
    # savings[i][j] = D[0][ci]+D[0][cj]-D[ci][cj]
    savings = []
    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = customers[i], customers[j]
            s = float(D[0, ci]) + float(D[0, cj]) - float(D[ci, cj])
            savings.append((s, i, j))
    savings.sort(reverse=True)
    in_route = {c: -1 for c in customers}
    for s, i, j in savings:
        if s <= 1e-9:
            break
        ci, cj = customers[i], customers[j]
        ri, rj = in_route[ci], in_route[cj]
        if ri == -1 or rj == -1 or ri == rj:
            continue
        # only merge if ci is at end of its route and cj at start of its route
        route_i = routes[ri]
        route_j = routes[rj]
        if route_i[-1] != ci or route_j[0] != cj:
            continue
        # capacity check
        if sum(float(demands[v]) for v in route_i) + sum(float(demands[v]) for v in route_j) > cap + 1e-9:
            continue
        merged = route_i + route_j
        if not _feasible(merged, D, demands, ready, due, service, battery, speed, rr, horizon):
            # try with a station
            fx = _fix_stations(merged, D, demands, ready, due, service, battery, speed, rr, horizon, stations)
            if fx is None:
                continue
            merged = fx
        routes[ri] = merged
        del routes[rj]
        for v in route_j:
            in_route[v] = ri
    return routes


def plan(instance):
    t0 = time.time()
    deadline = 110.0  # safety margin under 120s
    (D, demands, ready, due, service, n, n_cust, stations,
     cap, battery, speed, rr, horizon) = _build(instance)
    customers = list(range(1, n_cust + 1))

    # Build initial routes via Clarke-Wright
    routes = _clarke_wright(customers, D, demands, ready, due, service,
                            battery, speed, rr, horizon, stations, cap)

    def total_dist(routes):
        return sum(_dist_seq(r, D) for r in routes)

    best_routes = [list(r) for r in routes]
    best_score = total_dist(best_routes)

    def try_accept(routes):
        nonlocal best_routes, best_score
        s = total_dist(routes)
        if s < best_score - 1e-9:
            best_routes = [list(r) for r in routes]
            best_score = s
            return True
        return False

    # Local search loop
    while time.time() - t0 < deadline:
        progress = False
        # Intra-route 2-opt
        for ri in range(len(routes)):
            if time.time() - t0 > deadline:
                break
            routes[ri] = _two_opt(routes[ri], D, demands, ready, due, service,
                                  battery, speed, rr, horizon)
        # Intra-route Or-opt
        for ri in range(len(routes)):
            if time.time() - t0 > deadline:
                break
            routes[ri] = _or_opt(routes[ri], D, demands, ready, due, service,
                                 battery, speed, rr, horizon)
        # Inter-route swaps
        for ri in range(len(routes)):
            for rj in range(ri + 1, len(routes)):
                if time.time() - t0 > deadline:
                    break
                A = list(routes[ri])
                B = list(routes[rj])
                # try swapping each customer in A with each in B
                for ia, a in enumerate(A):
                    for ib, b in enumerate(B):
                        if service[a] <= 0 or service[b] <= 0:
                            continue
                        if a == b:
                            continue
                        A2 = list(A)
                        B2 = list(B)
                        A2[ia] = b
                        B2[ib] = a
                        okA = _feasible(A2, D, demands, ready, due, service, battery, speed, rr, horizon)
                        okB = _feasible(B2, D, demands, ready, due, service, battery, speed, rr, horizon)
                        if okA and okB:
                            newdist = _dist_seq(A2, D) + _dist_seq(B2, D)
                            olddist = _dist_seq(A, D) + _dist_seq(B, D)
                            if newdist < olddist - 1e-9:
                                routes[ri] = A2
                                routes[rj] = B2
                                progress = True
                                break
                    if progress:
                        break
                if progress:
                    break
            if progress:
                break
        # Inter-route relocate: move a customer from one route to another
        for ri in range(len(routes)):
            for rj in range(len(routes)):
                if ri == rj:
                    continue
                if time.time() - t0 > deadline:
                    break
                A = list(routes[ri])
                B = list(routes[rj])
                for ia, a in enumerate(A):
                    if service[a] <= 0:
                        continue
                    A2 = list(A)
                    A2.pop(ia)
                    bestpos = None
                    bestcost = None
                    for p in range(len(B) + 1):
                        B2 = B[:p] + [a] + B[p:]
                        r = _sim(B2, D, demands, ready, due, service, battery, speed, rr, horizon)
                        if r is None:
                            continue
                        cost = _dist_seq(A2, D) + _dist_seq(B2, D)
                        if bestcost is None or cost < bestcost:
                            bestcost = cost
                            bestpos = (p, B2)
                    if bestpos is not None:
                        oldcost = _dist_seq(A, D) + _dist_seq(B, D)
                        if bestcost < oldcost - 1e-9:
                            p, B2 = bestpos
                            routes[ri] = A2
                            routes[rj] = B2
                            progress = True
                            break
                if progress:
                    break
            if progress:
                break
        try_accept(routes)
        if not progress:
            # try a quick random perturbation to escape local minima
            if time.time() - t0 > deadline:
                break
            rng = np.random.default_rng(int(time.time() * 1000) % (2**31))
            perturbed = [list(r) for r in routes]
            # random 2-opt on a random route
            if perturbed:
                ri = int(rng.integers(0, len(perturbed)))
                r = perturbed[ri]
                if len(r) >= 3:
                    i = int(rng.integers(0, len(r) - 1))
                    j = int(rng.integers(i + 2, len(r)))
                    newr = r[:i] + list(reversed(r[i:j + 1])) + r[j + 1:]
                    if _feasible(newr, D, demands, ready, due, service, battery, speed, rr, horizon):
                        perturbed[ri] = newr
            # refine perturbed
            for k in range(len(perturbed)):
                perturbed[k] = _two_opt(perturbed[k], D, demands, ready, due, service,
                                        battery, speed, rr, horizon)
            s = total_dist(perturbed)
            if s < best_score + 5.0:  # accept slightly worse to explore
                routes = perturbed
            else:
                routes = [list(r) for r in best_routes]
            try_accept(routes)

    try_accept(best_routes)
    return best_routes
# EVOLVE-BLOCK-END
