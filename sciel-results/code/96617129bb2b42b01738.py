"""EV routing with time windows: cluster-first + nearest-neighbor + multi-neighborhood local search."""
# EVOLVE-BLOCK-START
import math
import time
import random
import heapq

_EPS = 1e-9


def _build(instance):
    coords = instance["coords"]
    n = len(coords)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = coords[i]
        row = D[i]
        for j in range(n):
            if i == j:
                continue
            xj, yj = coords[j]
            row[j] = math.hypot(xi - xj, yi - yj)
    return D


def _simulate(route, instance, D, stations_set):
    """Return (ok, total_distance)."""
    if not route:
        return True, 0.0
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge = instance["recharge_rate"]
    horizon = instance["horizon"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    demands = instance["demands"]

    load = 0.0
    battery = float(battery_max)
    t = 0.0
    prev = 0
    total = 0.0

    for node in route:
        d = D[prev][node]
        if d > battery + _EPS:
            return False, total
        battery -= d
        t += d / speed
        total += d
        prev = node
        if node in stations_set:
            if battery < 0.0:
                return False, total
            charge = battery_max - battery
            if charge > 0.0:
                t += charge * recharge
            battery = float(battery_max)
            continue
        load += demands[node]
        if load > capacity + _EPS:
            return False, total
        if t < ready[node]:
            t = float(ready[node])
        if t > due[node] + _EPS:
            return False, total
        t += service[node]
        if t > horizon + _EPS:
            return False, total

    d = D[prev][0]
    if d > battery + _EPS:
        return False, total
    battery -= d
    t += d / speed
    total += d
    if t > horizon + _EPS:
        return False, total
    return True, total


def _route_dist(route, D):
    if not route:
        return 0.0
    total = D[0][route[0]]
    for i in range(1, len(route)):
        total += D[route[i - 1]][route[i]]
    total += D[route[-1]][0]
    return total


def _total_dist(routes, D):
    return sum(_route_dist(r, D) for r in routes)


def _cluster_customers(instance, D):
    """Cluster customers using k-means-like approach based on distance from depot."""
    n_cust = instance["n_customers"]
    customers = list(range(1, n_cust + 1))
    capacity = instance["capacity"]
    demands = instance["demands"]

    total_demand = sum(demands[c] for c in customers)
    n_routes_est = max(1, int(math.ceil(total_demand / capacity)))

    random.seed(42)
    n_clusters = n_routes_est
    centers = random.sample(customers, min(n_clusters, len(customers)))

    clusters = [[] for _ in range(n_clusters)]
    for _ in range(30):
        new_clusters = [[] for _ in range(n_clusters)]
        for c in customers:
            best_center = 0
            best_dist = float('inf')
            for k, center in enumerate(centers):
                d = D[c][center]
                if d < best_dist:
                    best_dist = d
                    best_center = k
            new_clusters[best_center].append(c)

        if all(len(new_clusters[k]) == len(clusters[k]) for k in range(n_clusters)):
            break
        clusters = new_clusters

        for k in range(n_clusters):
            if clusters[k]:
                if len(clusters[k]) == 1:
                    centers[k] = clusters[k][0]
                else:
                    # Pick the one closest to centroid
                    cx = sum(instance["coords"][c][0] for c in clusters[k]) / len(clusters[k])
                    cy = sum(instance["coords"][c][1] for c in clusters[k]) / len(clusters[k])
                    best_c = clusters[k][0]
                    best_d = float('inf')
                    for c in clusters[k]:
                        d = math.hypot(instance["coords"][c][0] - cx, instance["coords"][c][1] - cy)
                        if d < best_d:
                            best_d = d
                            best_c = c
                    centers[k] = best_c

    final_clusters = [c for c in clusters if c]

    result = []
    for cluster in final_clusters:
        if not cluster:
            continue
        sub_clusters = []
        current = []
        current_demand = 0
        # Sort by angle from depot for better clustering
        def angle(c):
            x, y = instance["coords"][c]
            return math.atan2(y, x)
        cluster_sorted = sorted(cluster, key=angle)
        for c in cluster_sorted:
            if current_demand + demands[c] > capacity:
                if current:
                    sub_clusters.append(current)
                current = [c]
                current_demand = demands[c]
            else:
                current.append(c)
                current_demand += demands[c]
        if current:
            sub_clusters.append(current)
        result.extend(sub_clusters)

    return result


def _nn_route(cluster, instance, D, stations_set):
    """Build a route for a cluster using nearest neighbor with time window awareness."""
    if not cluster:
        return []

    remaining = set(cluster)
    route = []
    current = 0
    battery = float(instance["battery"])
    t = 0.0
    load = 0.0
    capacity = instance["capacity"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    demands = instance["demands"]
    speed = instance["speed"]
    recharge = instance["recharge_rate"]
    battery_max = instance["battery"]

    while remaining:
        best_node = None
        best_score = float('inf')
        best_station = None

        for c in remaining:
            d = D[current][c]
            if d > battery + _EPS:
                continue
            new_t = t + d / speed
            if new_t < ready[c]:
                new_t = float(ready[c])
            if new_t > due[c] + _EPS:
                continue
            new_load = load + demands[c]
            if new_load > capacity + _EPS:
                continue
            score = d + 0.1 * (new_t - ready[c])
            if score < best_score:
                best_score = score
                best_node = c
                best_station = None

        if best_node is None:
            found = False
            for s in instance["stations"]:
                d_to_s = D[current][s]
                if d_to_s > battery + _EPS:
                    continue
                new_battery = battery - d_to_s
                new_t = t + d_to_s / speed
                charge = battery_max - new_battery
                if charge > 0:
                    new_t += charge * recharge
                new_battery = battery_max

                for c in remaining:
                    d_s_c = D[s][c]
                    if d_s_c > new_battery + _EPS:
                        continue
                    new_t2 = new_t + d_s_c / speed
                    if new_t2 < ready[c]:
                        new_t2 = float(ready[c])
                    if new_t2 > due[c] + _EPS:
                        continue
                    new_load = load + demands[c]
                    if new_load > capacity + _EPS:
                        continue
                    score = d_to_s + d_s_c + 0.1 * (new_t2 - ready[c])
                    if score < best_score:
                        best_score = score
                        best_node = c
                        best_station = s
                        found = True
                        break
                if found:
                    break
            if not found:
                for c in remaining:
                    d = D[current][c]
                    if d > battery + _EPS:
                        continue
                    new_t = t + d / speed
                    if new_t < ready[c]:
                        new_t = float(ready[c])
                    if new_t > due[c] + _EPS:
                        continue
                    new_load = load + demands[c]
                    if new_load > capacity + _EPS:
                        continue
                    best_node = c
                    best_station = None
                    break

        if best_node is None:
            break

        if best_station is not None:
            d = D[current][best_station]
            battery -= d
            t += d / speed
            charge = battery_max - battery
            if charge > 0:
                t += charge * recharge
            battery = battery_max
            route.append(best_station)
            current = best_station
            d = D[current][best_node]
            battery -= d
            t += d / speed
            load += demands[best_node]
            if t < ready[best_node]:
                t = float(ready[best_node])
            t += service[best_node]
            route.append(best_node)
            remaining.discard(best_node)
            current = best_node
        else:
            d = D[current][best_node]
            battery -= d
            t += d / speed
            load += demands[best_node]
            if t < ready[best_node]:
                t = float(ready[best_node])
            t += service[best_node]
            route.append(best_node)
            remaining.discard(best_node)
            current = best_node

    return route


def _two_opt(route, instance, D, stations_set, deadline):
    """2-opt within a route."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        m = len(route)
        if m < 3:
            break
        old_dist = _route_dist(route, D)
        for i in range(m - 1):
            if time.time() >= deadline:
                break
            for j in range(i + 1, m):
                cand = route[:i] + route[i:j+1][::-1] + route[j+1:]
                ok, _ = _simulate(cand, instance, D, stations_set)
                if ok:
                    new_dist = _route_dist(cand, D)
                    if new_dist < old_dist - _EPS:
                        route = cand
                        old_dist = new_dist
                        improved = True
                        break
            if improved:
                break
    return route


def _or_opt(route, instance, D, stations_set, deadline):
    """Or-opt within a route: move 1-3 consecutive nodes."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        m = len(route)
        if m < 3:
            break
        old_dist = _route_dist(route, D)
        for seg_len in (1, 2, 3):
            if m < seg_len + 1:
                continue
            for i in range(m - seg_len + 1):
                if time.time() >= deadline:
                    break
                seg = route[i:i+seg_len]
                if any(n in stations_set for n in seg):
                    continue
                rest = route[:i] + route[i+seg_len:]
                for j in range(len(rest) + 1):
                    if j == i or j == i + seg_len:
                        continue
                    cand = rest[:j] + seg + rest[j:]
                    ok, _ = _simulate(cand, instance, D, stations_set)
                    if ok:
                        new_dist = _route_dist(cand, D)
                        if new_dist < old_dist - _EPS:
                            route = cand
                            old_dist = new_dist
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    return route


def _cross_route_2opt(routes, instance, D, stations_set, deadline):
    """2-opt between two routes: relocate a customer from one route to another."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        nr = len(routes)
        for ri in range(nr):
            for rj in range(nr):
                if ri == rj:
                    continue
                if time.time() >= deadline:
                    return routes
                ra = routes[ri]
                rb = routes[rj]
                if not ra or not rb:
                    continue
                # Try moving each customer from ra to rb
                for i in range(len(ra)):
                    if ra[i] in stations_set:
                        continue
                    c = ra[i]
                    ra_new = ra[:i] + ra[i+1:]
                    for j in range(len(rb) + 1):
                        if j == 0 or rb[j-1] in stations_set or (j < len(rb) and rb[j] in stations_set):
                            continue
                        rb_new = rb[:j] + [c] + rb[j:]
                        ok1, _ = _simulate(ra_new, instance, D, stations_set)
                        ok2, _ = _simulate(rb_new, instance, D, stations_set)
                        if ok1 and ok2:
                            old_dist = _route_dist(ra, D) + _route_dist(rb, D)
                            new_dist = _route_dist(ra_new, D) + _route_dist(rb_new, D)
                            if new_dist < old_dist - _EPS:
                                routes[ri] = ra_new
                                routes[rj] = rb_new
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        if not improved:
            break
    return routes


def _local_search(routes, instance, D, stations_set, deadline):
    """Apply local search to all routes."""
    # First pass: 2-opt on each route
    for ri in range(len(routes)):
        if time.time() >= deadline:
            break
        routes[ri] = _two_opt(routes[ri], instance, D, stations_set, deadline)

    # Second pass: Or-opt on each route
    for ri in range(len(routes)):
        if time.time() >= deadline:
            break
        routes[ri] = _or_opt(routes[ri], instance, D, stations_set, deadline)

    # Third pass: cross-route moves
    if time.time() < deadline - 5:
        routes = _cross_route_2opt(routes, instance, D, stations_set, deadline)

    # Fourth pass: 2-opt again
    for ri in range(len(routes)):
        if time.time() >= deadline:
            break
        routes[ri] = _two_opt(routes[ri], instance, D, stations_set, deadline)

    # Fifth pass: Or-opt again
    for ri in range(len(routes)):
        if time.time() >= deadline:
            break
        routes[ri] = _or_opt(routes[ri], instance, D, stations_set, deadline)

    # Remove empty routes
    routes = [r for r in routes if r]

    return routes


def _routes_valid(routes, instance, D, stations_set):
    seen = set()
    for r in routes:
        ok, _ = _simulate(r, instance, D, stations_set)
        if not ok:
            return False
        for node in r:
            if node in stations_set or node == 0:
                continue
            if node in seen:
                return False
            seen.add(node)
    n_cust = instance["n_customers"]
    if len(seen) != n_cust:
        return False
    return True


def plan(instance):
    t0 = time.time()
    deadline = t0 + 140.0
    stations_set = set(instance["stations"])
    D = _build(instance)

    clusters = _cluster_customers(instance, D)

    routes = []
    for cluster in clusters:
        if time.time() >= deadline:
            break
        route = _nn_route(cluster, instance, D, stations_set)
        if route:
            routes.append(route)

    covered = set()
    for r in routes:
        for node in r:
            if node not in stations_set and node != 0:
                covered.add(node)
    missing = [v for v in range(1, instance["n_customers"] + 1) if v not in covered]
    for v in missing:
        if time.time() >= deadline:
            break
        inserted = False
        for ri in range(len(routes)):
            r = routes[ri]
            for pos in range(len(r) + 1):
                cand = r[:pos] + [v] + r[pos:]
                ok, _ = _simulate(cand, instance, D, stations_set)
                if ok:
                    routes[ri] = cand
                    inserted = True
                    break
            if inserted:
                break
        if not inserted:
            routes.append([v])

    if time.time() < deadline - 5:
        routes = _local_search(routes, instance, D, stations_set, deadline)

    if not _routes_valid(routes, instance, D, stations_set):
        routes = [[v] for v in range(1, instance["n_customers"] + 1)]

    return routes
# EVOLVE-BLOCK-END
