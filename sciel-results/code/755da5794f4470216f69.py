"""EV routing with time windows and charging stations."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time
from math import sqrt, inf

def _euclid(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return sqrt(dx*dx + dy*dy)

def _kmeans(points, k, seed=0, iters=30):
    rng = np.random.default_rng(seed)
    n = len(points)
    if k >= n:
        return list(range(n))
    pts = np.array(points, dtype=float)
    # init: pick k distinct points
    idx = rng.choice(n, size=k, replace=False)
    centers = pts[idx].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        d = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(d, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            m = labels == c
            if m.any():
                centers[c] = pts[m].mean(axis=0)
    return labels.tolist()

def _simulate(route_nodes, coords, demands, ready, due, service,
              stations, battery_cap, recharge_rate, horizon, speed=1.0):
    """Return (feasible, total_distance, end_time)."""
    t = 0.0
    batt = battery_cap
    dist = 0.0
    prev = 0
    for node in route_nodes:
        d = _euclid(coords[prev], coords[node])
        dist += d
        t += d / speed
        batt -= d
        if batt < -1e-9:
            return False, dist, t
        if node in stations:
            # recharge
            t += (batt - 0.0) * recharge_rate
            batt = battery_cap
        else:
            # customer: time window + service
            if t < ready[node]:
                t = ready[node]
            if t > due[node] + 1e-9:
                return False, dist, t
            t += service[node]
            batt -= 0  # service doesn't drain battery
        prev = node
    # return to depot
    d = _euclid(coords[prev], coords[0])
    dist += d
    t += d / speed
    batt -= d
    if batt < -1e-9 or t > horizon + 1e-9:
        return False, dist, t
    return True, dist, t

def _route_distance(route_nodes, coords):
    d = 0.0
    prev = 0
    for node in route_nodes:
        d += _euclid(coords[prev], coords[node])
        prev = node
    d += _euclid(coords[prev], coords[0])
    return d

def _try_insert_charging(route, node, coords, demands, ready, due, service,
                         stations, battery_cap, recharge_rate, horizon, capacity):
    """Try to insert customer `node` into `route` (list of node ids, no depot).
    Returns new route list if feasible (with charging inserted as needed), else None.
    We try each position; if battery runs low, insert a charging station visit."""
    # First check demand
    total_demand = sum(demands[v] for v in route) + demands[node]
    if total_demand > capacity + 1e-9:
        return None

    # Try inserting at each position without charging first
    for pos in range(len(route) + 1):
        trial = route[:pos] + [node] + route[pos:]
        feasible, dist, end_t = _simulate(trial, coords, demands, ready, due,
                                           service, stations, battery_cap,
                                           recharge_rate, horizon)
        if feasible:
            return trial

    # Try inserting with a charging station before or after the new customer
    for pos in range(len(route) + 1):
        for st in stations:
            # insert customer then station
            trial = route[:pos] + [node, st] + route[pos:]
            feasible, dist, end_t = _simulate(trial, coords, demands, ready, due,
                                               service, stations, battery_cap,
                                               recharge_rate, horizon)
            if feasible:
                return trial
            # insert station then customer
            trial = route[:pos] + [st, node] + route[pos:]
            feasible, dist, end_t = _simulate(trial, coords, demands, ready, due,
                                               service, stations, battery_cap,
                                               recharge_rate, horizon)
            if feasible:
                return trial
    return None

def _construct_routes(instance):
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    customers = list(range(1, n_customers + 1))
    # Sort by due time (earliest first)
    customers.sort(key=lambda v: due[v])

    # Cluster into groups
    k = max(1, n_customers // 8)  # aim for ~8 customers per route
    k = min(k, n_customers)
    pts = [coords[v] for v in customers]
    labels = _kmeans(pts, k, seed=42)

    # Build clusters
    clusters = [[] for _ in range(k)]
    for i, v in enumerate(customers):
        clusters[labels[i]].append(v)

    routes = []
    for cl in clusters:
        if not cl:
            continue
        # Sort cluster by nearest neighbor from depot
        route = []
        remaining = list(cl)
        prev = 0
        while remaining:
            best = None
            best_d = inf
            for v in remaining:
                d = _euclid(coords[prev], coords[v])
                if d < best_d:
                    best_d = d
                    best = v
            # Try to insert
            new_route = _try_insert_charging(route, best, coords, demands, ready,
                                              due, service, stations, battery_cap,
                                              recharge_rate, horizon, capacity)
            if new_route is None:
                # Can't insert; start new route
                if route:
                    routes.append(route)
                route = []
                prev = 0
                new_route = _try_insert_charging(route, best, coords, demands, ready,
                                                  due, service, stations, battery_cap,
                                                  recharge_rate, horizon, capacity)
                if new_route is None:
                    # Fallback: single customer route
                    route = [best]
                else:
                    route = new_route
            else:
                route = new_route
            remaining.remove(best)
            prev = best
        if route:
            routes.append(route)

    # Check all customers covered
    covered = set()
    for r in routes:
        for v in r:
            if v not in stations:
                covered.add(v)
    missing = set(range(1, n_customers + 1)) - covered
    for v in missing:
        # Add as separate route
        routes.append([v])

    return routes

def _local_search(routes, instance, max_iters=200, deadline=None):
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    def total_dist(rts):
        return sum(_route_distance(r, coords) for r in rts)

    def feasible_routes(rts):
        for r in rts:
            f, _, _ = _simulate(r, coords, demands, ready, due, service,
                                stations, battery_cap, recharge_rate, horizon)
            if not f:
                return False
        return True

    best_routes = [list(r) for r in routes]
    best_dist = total_dist(best_routes)
    if not feasible_routes(best_routes):
        return best_routes

    for it in range(max_iters):
        if deadline and _time.time() > deadline:
            break
        improved = False

        # 2-opt within each route (customers only, skip stations for simplicity)
        for ri in range(len(best_routes)):
            r = best_routes[ri]
            # Extract customer positions
            cust_positions = [i for i, v in enumerate(r) if v not in stations]
            for ai in range(len(cust_positions)):
                for bi in range(ai + 2, len(cust_positions)):
                    i, j = cust_positions[ai], cust_positions[bi]
                    # Reverse segment between i and j
                    new_r = r[:i] + r[i:j+1][::-1] + r[j+1:]
                    if _simulate(new_r, coords, demands, ready, due, service,
                                 stations, battery_cap, recharge_rate, horizon)[0]:
                        nd = _route_distance(new_r, coords)
                        old_d = _route_distance(r, coords)
                        if nd < old_d - 1e-9:
                            best_routes[ri] = new_r
                            best_dist += nd - old_d
                            improved = True
                            r = new_r
                            cust_positions = [i for i, v in enumerate(r) if v not in stations]

        # Relocate: move a customer from one route to another
        for ri in range(len(best_routes)):
            for rj in range(len(best_routes)):
                if ri == rj:
                    continue
                r = list(best_routes[ri])
                for idx, v in enumerate(r):
                    if v in stations:
                        continue
                    # Try removing v from ri
                    r_new = r[:idx] + r[idx+1:]
                    if not r_new:
                        continue
                    if not _simulate(r_new, coords, demands, ready, due, service,
                                     stations, battery_cap, recharge_rate, horizon)[0]:
                        continue
                    # Try inserting v into rj
                    rj_new = _try_insert_charging(list(best_routes[rj]), v, coords,
                                                   demands, ready, due, service,
                                                   stations, battery_cap,
                                                   recharge_rate, horizon, capacity)
                    if rj_new is None:
                        continue
                    # Check total distance
                    old_d = _route_distance(r, coords) + _route_distance(best_routes[rj], coords)
                    new_d = _route_distance(r_new, coords) + _route_distance(rj_new, coords)
                    if new_d < old_d - 1e-9:
                        best_routes[ri] = r_new
                        best_routes[rj] = rj_new
                        best_dist += new_d - old_d
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

        if not improved:
            break

    return best_routes

def plan(instance):
    start_time = _time.time()
    deadline = start_time + 140.0  # safety margin

    # Initial construction
    routes = _construct_routes(instance)

    # Local search
    routes = _local_search(routes, instance, max_iters=300, deadline=deadline)

    # Verify feasibility; if not, fall back to single-customer routes
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = instance["stations"]
    battery_cap = instance["battery"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    def all_feasible(rts):
        for r in rts:
            f, _, _ = _simulate(r, coords, demands, ready, due, service,
                                stations, battery_cap, recharge_rate, horizon)
            if not f:
                return False
        # Check all customers covered exactly once
        seen = set()
        for r in rts:
            for v in r:
                if v not in stations:
                    if v in seen:
                        return False
                    seen.add(v)
        expected = set(range(1, instance["n_customers"] + 1))
        return seen == expected

    if not all_feasible(routes):
        # Fallback: one customer per route
        customers = list(range(1, instance["n_customers"] + 1))
        customers.sort(key=lambda v: due[v])
        routes = [[v] for v in customers]

    return routes
# EVOLVE-BLOCK-END
