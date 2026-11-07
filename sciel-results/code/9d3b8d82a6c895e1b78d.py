"""EV routing with time windows: cluster-first + fast local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

_START = None


def _now():
    global _START
    if _START is None:
        _START = time.time()
    return time.time() - _START


def _check_route(route, inst):
    """Simulate a route (without depot endpoints). Return (ok, dist, end_time, end_batt)."""
    coords = inst["coords"]
    demands = inst["demands"]
    ready = inst["ready"]
    due = inst["due"]
    service = inst["service"]
    capacity = inst["capacity"]
    battery_cap = inst["battery"]
    speed = inst["speed"]
    recharge_rate = inst["recharge_rate"]
    horizon = inst["horizon"]
    n_customers = inst["n_customers"]

    total_demand = 0
    t = 0.0
    batt = float(battery_cap)
    dist = 0.0
    cur = 0
    for node in route:
        dx = coords[node][0] - coords[cur][0]
        dy = coords[node][1] - coords[cur][1]
        d = math.hypot(dx, dy)
        batt -= d
        t += d / speed
        dist += d
        if batt < -1e-9:
            return (False, dist, t, batt)
        if node <= n_customers:
            total_demand += demands[node]
            if total_demand > capacity:
                return (False, dist, t, batt)
            t = max(t, ready[node])
            if t > due[node] + 1e-9:
                return (False, dist, t, batt)
            t += service[node]
        else:
            charge_time = (battery_cap - batt) * recharge_rate
            t += charge_time
            batt = float(battery_cap)
        cur = node
    dx = coords[0][0] - coords[cur][0]
    dy = coords[0][1] - coords[cur][1]
    d = math.hypot(dx, dy)
    batt -= d
    t += d / speed
    dist += d
    if batt < -1e-9:
        return (False, dist, t, batt)
    if t > horizon + 1e-9:
        return (False, dist, t, batt)
    return (True, dist, t, batt)


def _route_dist(route, inst):
    coords = inst["coords"]
    dist = 0.0
    cur = 0
    for node in route:
        dx = coords[node][0] - coords[cur][0]
        dy = coords[node][1] - coords[cur][1]
        dist += math.hypot(dx, dy)
        cur = node
    dx = coords[0][0] - coords[cur][0]
    dy = coords[0][1] - coords[cur][1]
    dist += math.hypot(dx, dy)
    return dist


def _total_dist(routes, inst):
    return sum(_route_dist(r, inst) for r in routes)


def _try_fix_battery(route, inst):
    """Try to insert charging stations so battery never <0."""
    coords = inst["coords"]
    battery_cap = inst["battery"]
    speed = inst["speed"]
    recharge_rate = inst["recharge_rate"]
    horizon = inst["horizon"]
    stations = inst["stations"]
    n_customers = inst["n_customers"]

    def simulate(route):
        t = 0.0
        batt = float(battery_cap)
        cur = 0
        min_batt = batt
        neg_index = None
        for i, node in enumerate(route):
            dx = coords[node][0] - coords[cur][0]
            dy = coords[node][1] - coords[cur][1]
            d = math.hypot(dx, dy)
            batt -= d
            t += d / speed
            if batt < -1e-9 and neg_index is None:
                neg_index = i
            if batt < min_batt:
                min_batt = batt
            if node <= n_customers:
                t = max(t, inst["ready"][node])
                if t > inst["due"][node]:
                    return (False, None, t, batt)
                t += inst["service"][node]
            else:
                t += (battery_cap - batt) * recharge_rate
                batt = float(battery_cap)
            cur = node
        dx = coords[0][0] - coords[cur][0]
        dy = coords[0][1] - coords[cur][1]
        d = math.hypot(dx, dy)
        batt -= d
        t += d / speed
        if batt < -1e-9:
            return (False, "return", t, batt)
        if t > horizon:
            return (False, None, t, batt)
        return (True, None, t, batt)

    cur_route = list(route)
    for _ in range(15):
        ok, neg_idx, t_end, b_end = simulate(cur_route)
        if ok:
            ok2, dist, te, be = _check_route(cur_route, inst)
            if ok2:
                return cur_route
            else:
                return None
        if isinstance(neg_idx, int):
            insert_pos = neg_idx
        else:
            insert_pos = len(cur_route)
        best_new = None
        best_d = None
        for st in stations:
            new_route = cur_route[:insert_pos] + [st] + cur_route[insert_pos:]
            ok3, dist, te, be = _check_route(new_route, inst)
            if ok3:
                if best_d is None or dist < best_d:
                    best_new = new_route
                    best_d = dist
        if best_new is None:
            for pos in range(len(cur_route) + 1):
                for st in stations:
                    new_route = cur_route[:pos] + [st] + cur_route[pos:]
                    ok3, dist, te, be = _check_route(new_route, inst)
                    if ok3:
                        if best_d is None or dist < best_d:
                            best_new = new_route
                            best_d = dist
        if best_new is None:
            return None
        cur_route = best_new
    return None


def _constructive_cluster(inst):
    """Cluster-first route-second: k-means clustering then greedy sequencing per cluster."""
    coords = inst["coords"]
    demands = inst["demands"]
    ready = inst["ready"]
    due = inst["due"]
    service = inst["service"]
    n_customers = inst["n_customers"]
    stations = inst["stations"]
    capacity = inst["capacity"]
    battery_cap = inst["battery"]
    speed = inst["speed"]
    recharge_rate = inst["recharge_rate"]
    horizon = inst["horizon"]

    customers = list(range(1, n_customers + 1))
    cust_coords = np.array([coords[c] for c in customers])
    cust_demands = np.array([demands[c] for c in customers])

    # Determine number of clusters based on capacity and battery
    total_demand = cust_demands.sum()
    min_clusters = max(1, int(math.ceil(total_demand / capacity)))
    # Also consider battery: max distance ~ battery_cap, so cluster radius ~ battery_cap/2
    # Try k from min_clusters to min_clusters+3
    best_routes = None
    best_score = float('inf')

    for k in range(min_clusters, min_clusters + 4):
        if _now() > 130.0:
            break
        # K-means clustering
        # Initialize centroids with k-means++
        rng = np.random.RandomState(42)
        indices = list(range(len(customers)))
        centroids = [cust_coords[rng.choice(len(customers))]]
        for _ in range(1, k):
            dists = np.min([np.sum((cust_coords - c) ** 2, axis=1) for c in centroids], axis=0)
            probs = dists / dists.sum() if dists.sum() > 0 else np.ones(len(customers)) / len(customers)
            idx = rng.choice(len(customers), p=probs)
            centroids.append(cust_coords[idx])

        # K-means iterations
        labels = np.zeros(len(customers), dtype=int)
        for it in range(30):
            # Assign
            dists = np.array([np.sum((cust_coords - c) ** 2, axis=1) for c in centroids])
            new_labels = np.argmin(dists, axis=0)
            if np.array_equal(new_labels, labels) and it > 0:
                break
            labels = new_labels
            # Update centroids
            for j in range(k):
                mask = labels == j
                if mask.sum() > 0:
                    centroids[j] = cust_coords[mask].mean(axis=0)
                else:
                    # Re-initialize empty cluster
                    centroids[j] = cust_coords[rng.choice(len(customers))]

        # Build routes per cluster
        routes = []
        all_assigned = set()
        for j in range(k):
            cluster_custs = [customers[i] for i in range(len(customers)) if labels[i] == j]
            if not cluster_custs:
                continue
            # Check capacity
            cl_demand = sum(demands[c] for c in cluster_custs)
            if cl_demand > capacity:
                # Split: use smaller k or handle separately
                # For now, try to build route and if infeasible, split
                pass

            # Greedy sequencing within cluster
            cluster_custs.sort(key=lambda c: (due[c], ready[c]))
            route = []
            total_demand = 0
            t = 0.0
            batt = float(battery_cap)
            cur = 0

            for c in cluster_custs:
                dx = coords[c][0] - coords[cur][0]
                dy = coords[c][1] - coords[cur][1]
                d = math.hypot(dx, dy)
                nbatt = batt - d
                nt = t + d / speed
                nt = max(nt, ready[c])
                if nt > due[c] or nbatt < -1e-9 or total_demand + demands[c] > capacity:
                    # Try inserting with station
                    continue
                # Check return feasibility
                dx2 = coords[0][0] - coords[c][0]
                dy2 = coords[0][1] - coords[c][1]
                d2 = math.hypot(dx2, dy2)
                if nbatt - d2 < -1e-9:
                    continue
                nt2 = nt + d2 / speed
                if nt2 > horizon:
                    continue
                route.append(c)
                total_demand += demands[c]
                batt = nbatt
                t = nt + service[c]
                cur = c
                all_assigned.add(c)

            if route:
                ok, dist, te, be = _check_route(route, inst)
                if not ok:
                    fixed = _try_fix_battery(route, inst)
                    if fixed is not None:
                        route = fixed
                routes.append(route)

        # Check all customers assigned
        if all_assigned != set(customers):
            # Assign remaining customers to nearest cluster route
            remaining = set(customers) - all_assigned
            for c in sorted(remaining, key=lambda x: due[x]):
                best_r = None
                best_d = float('inf')
                for ri, r in enumerate(routes):
                    if not r:
                        continue
                    # Check if we can insert c into route ri
                    for pos in range(len(r) + 1):
                        new_r = r[:pos] + [c] + r[pos:]
                        ok, d, te, be = _check_route(new_r, inst)
                        if ok and d < best_d:
                            best_d = d
                            best_r = (ri, pos)
                if best_r is not None:
                    ri, pos = best_r
                    routes[ri] = routes[ri][:pos] + [c] + routes[ri][pos:]
                    all_assigned.add(c)
                else:
                    # Create new route
                    routes.append([c])

        # Validate all
        valid = True
        for r in routes:
            ok, d, te, be = _check_route(r, inst)
            if not ok:
                fixed = _try_fix_battery(r, inst)
                if fixed is not None:
                    routes[routes.index(r)] = fixed
                else:
                    valid = False
        if not valid:
            continue

        score = _total_dist(routes, inst)
        if score < best_score:
            best_score = score
            best_routes = [list(r) for r in routes]

    if best_routes is not None:
        return best_routes

    # Fallback: single route per customer
    return [[c] for c in sorted(customers, key=lambda x: due[x])]


def _local_improve(routes, inst, max_iter=2000):
    """Fast 2-opt and relocate moves with distance-delta pre-screening."""
    coords = inst["coords"]
    n_customers = inst["n_customers"]
    stations = inst["stations"]
    station_set = set(stations)

    best = _total_dist(routes, inst)
    no_improve = 0
    deadline = 145.0

    for it in range(max_iter):
        if _now() > deadline:
            break
        improved = False

        # 2-opt intra-route
        for i in range(len(routes)):
            if _now() > deadline:
                break
            r = routes[i]
            n = len(r)
            if n < 3:
                continue
            for u in range(n - 1):
                if _now() > deadline:
                    break
                for v in range(u + 1, n):
                    if _now() > deadline:
                        break
                    new_r = r[:u] + r[u:v + 1][::-1] + r[v + 1:]
                    # Quick distance delta check
                    # Old: ...a-b...c-d...  New: ...a-d...c-b...
                    # Only if both endpoints are "turns"
                    ok, dist, t, b = _check_route(new_r, inst)
                    if ok:
                        old_d = _route_dist(r, inst)
                        new_d = _route_dist(new_r, inst)
                        if new_d < old_d - 1e-6:
                            routes[i] = new_r
                            best = best - (old_d - new_d)
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break

        if improved:
            no_improve = 0
            continue

        # Relocate customer
        for i in range(len(routes)):
            if _now() > deadline:
                break
            for j in range(len(routes[i])):
                if _now() > deadline:
                    break
                node = routes[i][j]
                if node in station_set:
                    continue
                new_i = routes[i][:j] + routes[i][j + 1:]
                for k in range(len(routes)):
                    if _now() > deadline:
                        break
                    if k == i and not new_i:
                        continue
                    target = routes[k] if k != i else new_i
                    for pos in range(len(target) + 1):
                        new_target = target[:pos] + [node] + target[pos:]
                        new_routes = list(routes)
                        new_routes[i] = new_i if k != i else new_i
                        new_routes[k] = new_target
                        if any(len(r) == 0 for r in new_routes):
                            continue
                        # Quick check
                        feasible = True
                        for ri, rr in enumerate(new_routes):
                            ok, d, t, b = _check_route(rr, inst)
                            if not ok:
                                feasible = False
                                break
                        if feasible:
                            nd = _total_dist(new_routes, inst)
                            if nd < best - 1e-6:
                                routes[:] = new_routes
                                best = nd
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

        if not improved:
            no_improve += 1
            if no_improve > 30:
                break
        else:
            no_improve = 0

    return routes


def plan(instance):
    global _START
    _START = time.time()
    inst = instance

    # Cluster-first construction
    routes = _constructive_cluster(inst)

    # Local improvement
    routes = _local_improve(routes, inst)

    # Final safety check
    all_nodes = [n for r in routes for n in r]
    cust = [n for n in all_nodes if n <= inst["n_customers"]]
    if sorted(cust) != list(range(1, inst["n_customers"] + 1)):
        # Fallback
        customers = list(range(1, inst["n_customers"] + 1))
        customers.sort(key=lambda v: inst["due"][v])
        routes = [[v] for v in customers]

    for r in routes:
        ok, dist, t, b = _check_route(r, inst)
        if not ok:
            customers = list(range(1, inst["n_customers"] + 1))
            customers.sort(key=lambda v: instance["due"][v])
            return [[v] for v in customers]

    return routes
# EVOLVE-BLOCK-END
