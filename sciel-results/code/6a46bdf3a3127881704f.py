# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START = None


def _elapsed():
    global _START
    if _START is None:
        _START = _time.time()
    return _time.time() - _START


def plan(instance):
    global _START
    _START = _time.time()
    DEADLINE = 145.0  # safety margin under 150s

    coords = np.asarray(instance["coords"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    ready = np.asarray(instance["ready"], dtype=float)
    due = np.asarray(instance["due"], dtype=float)
    service = np.asarray(instance["service"], dtype=float)
    n_cust = int(instance["n_customers"])
    stations = set(instance["stations"])
    capacity = float(instance["capacity"])
    battery_max = float(instance["battery"])
    speed = float(instance["speed"])
    recharge_rate = float(instance["recharge_rate"])
    horizon = float(instance["horizon"])
    depot = 0

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))

    customers = list(range(1, n_cust + 1))

    def route_dist(route):
        if not route:
            return 0.0
        d = dist[depot, route[0]]
        for i in range(len(route) - 1):
            d += dist[route[i], route[i + 1]]
        d += dist[route[-1], depot]
        return d

    def check_route(route):
        if not route:
            return True, 0.0
        cur = depot
        t = 0.0
        bat = battery_max
        load = 0.0
        total_d = 0.0
        for node in route:
            d = dist[cur, node]
            t += d / speed
            bat -= d
            if bat < -1e-9 or t > horizon + 1e-9:
                return False, total_d + d
            total_d += d
            if node in stations:
                need = (battery_max - bat)
                t += need * recharge_rate
                bat = battery_max
            else:
                load += demands[node]
                if load > capacity + 1e-9:
                    return False, total_d
                t = max(t, ready[node])
                if t > due[node] + 1e-9:
                    return False, total_d
                t += service[node]
                if t > horizon + 1e-9:
                    return False, total_d
            cur = node
        d = dist[cur, depot]
        t += d / speed
        bat -= d
        if bat < -1e-9 or t > horizon + 1e-9:
            return False, total_d + d
        total_d += d
        return True, total_d

    def build_route(order):
        route = []
        load = 0.0
        cur = depot
        t = 0.0
        bat = battery_max
        remaining = list(order)
        while remaining:
            best = None
            best_d = None
            for c in remaining:
                d = dist[cur, c]
                new_t = t + d / speed
                new_bat = bat - d
                if new_bat < -1e-9:
                    continue
                new_load = load + demands[c]
                if new_load > capacity + 1e-9:
                    continue
                nt = max(new_t, ready[c])
                if nt > due[c] + 1e-9:
                    continue
                nt += service[c]
                if nt > horizon + 1e-9:
                    continue
                if best_d is None or d < best_d:
                    best = c
                    best_d = d
            if best is None:
                break
            d = dist[cur, best]
            t += d / speed
            bat -= d
            load += demands[best]
            t = max(t, ready[best])
            t += service[best]
            cur = best
            route.append(best)
            remaining.remove(best)
        return route

    def initial_solution():
        cust_arr = np.array(customers)
        pts = coords[cust_arr]
        total_demand = demands[1:n_cust + 1].sum()
        k = max(4, int(np.ceil(total_demand / capacity * 1.2)))
        rng = np.random.RandomState(7)
        idx = rng.permutation(len(pts))
        centers = pts[idx[:k]].copy()
        for _ in range(30):
            dmat = np.sqrt(((pts[:, None, :] - centers[None, :, :]) ** 2).sum(-1))
            assign = dmat.argmin(axis=1)
            new_centers = np.array([pts[assign == j].mean(axis=0) if (assign == j).any() else centers[j] for j in range(k)])
            if np.allclose(new_centers, centers, atol=1e-6):
                break
            centers = new_centers
        dmat = np.sqrt(((pts[:, None, :] - centers[None, :, :]) ** 2).sum(-1))
        assign = dmat.argmin(axis=1)
        routes = []
        for j in range(k):
            members = [cust_arr[i] for i in range(len(cust_arr)) if assign[i] == j]
            if not members:
                continue
            members.sort(key=lambda c: dist[depot, c])
            r = build_route(members)
            if r:
                routes.append(r)
        covered = set()
        for r in routes:
            covered.update(r)
        for c in customers:
            if c not in covered:
                routes.append([c])
        return routes

    def ensure_feasible(routes):
        new_routes = []
        for r in routes:
            ok, d = check_route(r)
            if ok:
                new_routes.append(r)
            else:
                for c in r:
                    new_routes.append([c])
        return new_routes

    routes = ensure_feasible(initial_solution())

    def total_distance(routes):
        return sum(route_dist(r) for r in routes)

    best_routes = [list(r) for r in routes]
    best_total = total_distance(best_routes)

    def relocate(routes):
        improved = False
        n = len(routes)
        for i in range(n):
            if not routes[i]:
                continue
            for j in range(len(routes[i])):
                c = routes[i][j]
                routes[i].pop(j)
                best_target = None
                best_add = None
                for k in range(len(routes)):
                    if k == i:
                        continue
                    for pos in range(len(routes[k]) + 1):
                        new_route = routes[k][:pos] + [c] + routes[k][pos:]
                        ok, d = check_route(new_route)
                        if ok:
                            prev = depot if pos == 0 else routes[k][pos - 1]
                            nxt = routes[k][pos] if pos < len(routes[k]) else depot
                            add = dist[prev, c] + dist[c, nxt] - dist[prev, nxt]
                            if best_add is None or add < best_add:
                                best_add = add
                                best_target = (k, pos)
                if best_target is not None:
                    k, pos = best_target
                    routes[k].insert(pos, c)
                    improved = True
                else:
                    routes[i].insert(j, c)
                if _elapsed() > DEADLINE:
                    return improved
        return improved

    def two_opt(routes):
        improved = False
        for i in range(len(routes)):
            r = routes[i]
            n = len(r)
            if n < 2:
                continue
            old_d = route_dist(r)
            base = total_distance(routes) - old_d
            for a in range(n - 1):
                for b in range(a + 1, n):
                    new_r = r[:a] + r[a:b + 1][::-1] + r[b + 1:]
                    ok, d = check_route(new_r)
                    if ok:
                        if base + d < base + old_d - 1e-9:
                            routes[i] = new_r
                            old_d = d
                            improved = True
                if _elapsed() > DEADLINE:
                    return improved
        return improved

    no_improve = 0
    while _elapsed() < DEADLINE:
        progress = False
        if relocate(routes):
            progress = True
        if _elapsed() < DEADLINE and two_opt(routes):
            progress = True
        cur_total = total_distance(routes)
        if cur_total < best_total - 1e-9:
            best_routes = [list(r) for r in routes]
            best_total = cur_total
            no_improve = 0
        else:
            no_improve += 1
            if not progress:
                if no_improve > 5:
                    for _ in range(10):
                        i = int(np.random.randint(len(routes)))
                        r = routes[i]
                        if len(r) < 2:
                            continue
                        a = int(np.random.randint(len(r)))
                        b = int(np.random.randint(len(r)))
                        if a > b:
                            a, b = b, a
                        new_r = r[:a] + r[a:b + 1][::-1] + r[b + 1:]
                        ok, d = check_route(new_r)
                        if ok:
                            routes[i] = new_r
                    no_improve = 0
                else:
                    break

    final = []
    for r in best_routes:
        ok, d = check_route(r)
        if ok:
            final.append(r)
        else:
            for c in r:
                final.append([c])

    seen = set()
    clean = []
    for r in final:
        nr = []
        for c in r:
            if c not in seen:
                seen.add(c)
                nr.append(c)
        if nr:
            clean.append(nr)
    for c in customers:
        if c not in seen:
            clean.append([c])

    return clean
# EVOLVE-BLOCK-END
