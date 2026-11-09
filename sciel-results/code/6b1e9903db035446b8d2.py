"""EV routing with time windows and battery constraints."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time
from itertools import permutations


def _euclid(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _build(instance):
    coords = instance["coords"]
    n = len(coords)
    # Precompute distance matrix
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = _euclid(coords[i], coords[j])
    return dist, coords, n


def _route_eval(seq, dist, instance):
    """
    Evaluate a single route (list of node ids, depot implicit at both ends).
    Returns (total_distance, feasible_bool).
    seq should contain only customers and possibly stations (not depot 0).
    """
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = set(instance["stations"])

    total_dist = 0.0
    total_demand = 0.0
    battery = battery_max
    t = 0.0
    cur = 0  # depot

    for node in seq:
        d = dist[cur][node]
        total_dist += d
        battery -= d
        if battery < -1e-9:
            return total_dist, False
        t += d / speed
        cur = node
        if node in stations:
            # Recharge time
            recharge_time = (battery_max - battery) * recharge_rate
            t += recharge_time
            battery = battery_max
        else:
            # Customer
            total_demand += demands[node]
            if total_demand > capacity + 1e-9:
                return total_dist, False
            if t < ready[node]:
                t = ready[node]
            if t > due[node] + 1e-9:
                return total_dist, False
            t += service[node]

    # Return to depot
    d = dist[cur][0]
    total_dist += d
    battery -= d
    if battery < -1e-9:
        return total_dist, False
    t += d / speed
    if t > horizon + 1e-9:
        return total_dist, False

    return total_dist, True


def _full_eval(routes, dist, instance):
    """Evaluate all routes. Returns (total_distance, feasible)."""
    total_dist = 0.0
    for r in routes:
        d, ok = _route_eval(r, dist, instance)
        if not ok:
            return total_dist, False
        total_dist += d
    return total_dist, True


def _greedy_cluster(instance, dist, n, n_customers, stations, demands, n_routes=4):
    """
    Greedy: assign customers to n_routes clusters based on angle from depot,
    then order each cluster by nearest neighbor.
    """
    coords = instance["coords"]
    # Angle from depot for each customer
    angles = []
    for i in range(1, n_customers + 1):
        dx = coords[i][0] - coords[0][0]
        dy = coords[i][1] - coords[0][1]
        angles.append((np.arctan2(dy, dx), i))
    angles.sort()

    # Assign to n_routes buckets by angle
    buckets = [[] for _ in range(n_routes)]
    for idx, (ang, cust) in enumerate(angles):
        b = idx % n_routes
        buckets[b].append(cust)

    # For each bucket, order by nearest neighbor starting from depot
    routes = []
    for b in range(n_routes):
        if not buckets[b]:
            continue
        route = []
        remaining = set(buckets[b])
        cur = 0
        while remaining:
            best = min(remaining, key=lambda c: dist[cur][c])
            route.append(best)
            remaining.remove(best)
            cur = best
        routes.append(route)
    return routes


def _greedy_due(instance, dist, n, n_customers, stations, demands, n_routes=4):
    """
    Greedy: sort by due time, assign to the route that can accommodate with least extra distance.
    """
    customers = list(range(1, n_customers + 1))
    customers.sort(key=lambda v: instance["due"][v])

    routes = [[] for _ in range(n_routes)]
    route_dists = [0.0] * n_routes
    route_batteries = [instance["battery"]] * n_routes
    route_times = [0.0] * n_routes
    route_cur = [0] * n_routes  # last node
    route_demand = [0.0] * n_routes

    for cust in customers:
        best_r = -1
        best_cost = float('inf')
        for r in range(n_routes):
            if route_demand[r] + demands[cust] > instance["capacity"] + 1e-9:
                continue
            # Try appending cust to end of route r
            d = dist[route_cur[r]][cust]
            if route_batteries[r] - d < -1e-9:
                continue
            new_t = route_times[r] + d / instance["speed"]
            if new_t < instance["ready"][cust]:
                new_t = instance["ready"][cust]
            if new_t > instance["due"][cust] + 1e-9:
                continue
            new_t += instance["service"][cust]
            # Check return to depot
            d_ret = dist[cust][0]
            if (route_batteries[r] - d) - d_ret < -1e-9:
                continue
            new_t_ret = new_t + d_ret / instance["speed"]
            if new_t_ret > instance["horizon"] + 1e-9:
                continue
            cost = d
            if cost < best_cost:
                best_cost = cost
                best_r = r
        if best_r >= 0:
            r = best_r
            d = dist[route_cur[r]][cust]
            route_dists[r] += d
            route_batteries[r] -= d
            new_t = route_times[r] + d / instance["speed"]
            if new_t < instance["ready"][cust]:
                new_t = instance["ready"][cust]
            new_t += instance["service"][cust]
            route_times[r] = new_t
            route_cur[r] = cust
            route_demand[r] += demands[cust]
            routes[r].append(cust)
        else:
            # Fallback: put in first route
            routes[0].append(cust)

    # Remove empty routes
    return [r for r in routes if r]


def _local_search(routes, dist, instance, max_iter=2000, deadline=None):
    """
    Local search with 2-opt, relocate, and swap moves within and across routes.
    """
    n_customers = instance["n_customers"]
    all_customers = set(range(1, n_customers + 1))
    best_routes = [list(r) for r in routes]
    best_dist, best_ok = _full_eval(best_routes, dist, instance)
    if not best_ok:
        return routes  # infeasible, return as-is

    cur_routes = [list(r) for r in routes]
    cur_dist, cur_ok = _full_eval(cur_routes, dist, instance)
    if not cur_ok:
        return best_routes

    n = len(dist)
    # Build route index: node -> (route_idx, pos)
    def build_index(routes_):
        idx = {}
        for ri, r in enumerate(routes_):
            for pi, node in enumerate(r):
                idx[node] = (ri, pi)
        return idx

    def apply_move(routes_, move_type, args):
        """Apply a move and return new routes (copy)."""
        new = [list(r) for r in routes_]
        if move_type == '2opt':
            ri, i, j = args
            r = new[ri]
            # Reverse segment i..j
            seg = r[i:j+1]
            seg.reverse()
            r[i:j+1] = seg
        elif move_type == 'relocate':
            src_r, pos, dst_r = args
            node = new[src_r].pop(pos)
            new[dst_r].append(node)
        elif move_type == 'swap':
            ri1, pi1, ri2, pi2 = args
            n1 = new[ri1][pi1]
            n2 = new[ri2][pi2]
            new[ri1][pi1] = n2
            new[ri2][pi2] = n1
        return new

    idx = build_index(cur_routes)
    for it in range(max_iter):
        if deadline is not None and _time.time() > deadline:
            break
        improved = False
        # Try 2-opt within each route
        for ri in range(len(cur_routes)):
            r = cur_routes[ri]
            L = len(r)
            if L < 2:
                continue
            for i in range(L - 1):
                for j in range(i + 1, L):
                    new_routes = apply_move(cur_routes, '2opt', (ri, i, j))
                    # Check if any route is empty
                    if any(len(nr) == 0 for nr in new_routes):
                        continue
                    d, ok = _full_eval(new_routes, dist, instance)
                    if ok and d < cur_dist - 1e-9:
                        cur_routes = new_routes
                        cur_dist = d
                        idx = build_index(cur_routes)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue

        # Try relocate: move a node to another route (or same route different pos)
        for ri in range(len(cur_routes)):
            for pi in range(len(cur_routes[ri])):
                node = cur_routes[ri][pi]
                for dst_r in range(len(cur_routes)):
                    if dst_r == ri:
                        continue
                    new_routes = apply_move(cur_routes, 'relocate', (ri, pi, dst_r))
                    if any(len(nr) == 0 for nr in new_routes):
                        continue
                    d, ok = _full_eval(new_routes, dist, instance)
                    if ok and d < cur_dist - 1e-9:
                        cur_routes = new_routes
                        cur_dist = d
                        idx = build_index(cur_routes)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue

        # Try swap between two routes
        for ri1 in range(len(cur_routes)):
            for ri2 in range(ri1 + 1, len(cur_routes)):
                for pi1 in range(len(cur_routes[ri1])):
                    for pi2 in range(len(cur_routes[ri2])):
                        new_routes = apply_move(cur_routes, 'swap', (ri1, pi1, ri2, pi2))
                        d, ok = _full_eval(new_routes, dist, instance)
                        if ok and d < cur_dist - 1e-9:
                            cur_routes = new_routes
                            cur_dist = d
                            idx = build_index(cur_routes)
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

    if cur_dist < best_dist:
        best_routes = cur_routes
        best_dist = cur_dist

    return best_routes


def plan(instance):
    start_time = _time.time()
    deadline = start_time + 130.0  # 20s safety margin

    coords = instance["coords"]
    n = len(coords)
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    demands = instance["demands"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    dist, coords, n = _build(instance)

    # Generate multiple initial solutions
    inits = []

    # 1. Angle-based clustering
    r1 = _greedy_cluster(instance, dist, n, n_customers, stations, demands, n_routes=4)
    inits.append(r1)

    # 2. Due-time greedy
    r2 = _greedy_due(instance, dist, n, n_customers, stations, demands, n_routes=4)
    inits.append(r2)

    # 3. Random-ish: sort by x then y
    custs = list(range(1, n_customers + 1))
    custs.sort(key=lambda c: (coords[c][0], coords[c][1]))
    n_r = 4
    chunk = len(custs) // n_r + 1
    r3 = [custs[i*chunk:(i+1)*chunk] for i in range(n_r) if custs[i*chunk:(i+1)*chunk]]
    inits.append(r3)

    # Evaluate all inits, pick best feasible
    best_init = None
    best_init_dist = float('inf')
    for r in inits:
        d, ok = _full_eval(r, dist, instance)
        if ok and d < best_init_dist:
            best_init_dist = d
            best_init = r

    if best_init is None:
        # Fallback: one per route
        return [[v] for v in range(1, n_customers + 1)]

    # Local search
    best_routes = _local_search(best_init, dist, instance, max_iter=5000, deadline=deadline)

    # If we have time, try local search from other inits
    if _time.time() < deadline - 10.0:
        for r in inits:
            if r is best_init:
                continue
            d, ok = _full_eval(r, dist, instance)
            if not ok:
                continue
            if d >= best_init_dist:
                continue
            candidate = _local_search(r, dist, instance, max_iter=3000, deadline=deadline)
            cd, cok = _full_eval(candidate, dist, instance)
            if cok and cd < best_init_dist:
                best_init_dist = cd
                best_routes = candidate
            if _time.time() > deadline - 5.0:
                break

    # Final check
    d, ok = _full_eval(best_routes, dist, instance)
    if not ok:
        # Fallback
        return [[v] for v in range(1, n_customers + 1)]

    return best_routes
# EVOLVE-BLOCK-END
