"""EV routing: savings-based construction + battery-aware local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random

DEADLINE_SECONDS = 115.0


def _build_dist_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
            D[i][j] = d
            D[j][i] = d
    return D


def _validate_route(route, D, demands, ready, due, service, capacity,
                    battery_max, speed, recharge_rate, horizon, stations_set, n_customers):
    """Validate a single route. Returns (valid, distance)."""
    total_demand = 0
    for node in route:
        if 1 <= node <= n_customers:
            total_demand += demands[node]
    if total_demand > capacity:
        return False, 0.0

    cur = 0
    t = 0.0
    battery = battery_max
    dist = 0.0

    for node in route:
        d = D[cur][node]
        battery -= d
        t += d / speed
        if battery < -1e-9:
            return False, 0.0
        dist += d
        cur = node

        if node in stations_set:
            t += (battery_max - battery) * recharge_rate
            battery = battery_max
        else:
            t = max(t, ready[node])
            if t > due[node] + 1e-9:
                return False, 0.0
            t += service[node]

    d = D[cur][0]
    battery -= d
    t += d / speed
    dist += d
    if battery < -1e-9:
        return False, 0.0
    if t > horizon + 1e-9:
        return False, 0.0

    return True, dist


def _validate_solution(routes, D, demands, ready, due, service, capacity,
                       battery_max, speed, recharge_rate, horizon, stations_set, n_customers):
    all_visited = []
    for route in routes:
        for node in route:
            if 1 <= node <= n_customers:
                all_visited.append(node)
    if len(all_visited) != n_customers:
        return False, 0.0
    if len(set(all_visited)) != n_customers:
        return False, 0.0

    total_dist = 0.0
    for route in routes:
        valid, d = _validate_route(route, D, demands, ready, due, service,
                                   capacity, battery_max, speed, recharge_rate,
                                   horizon, stations_set, n_customers)
        if not valid:
            return False, 0.0
        total_dist += d
    return True, total_dist


def _compute_distance(route, D):
    if not route:
        return 0.0
    cur = 0
    total = 0.0
    for node in route:
        total += D[cur][node]
        cur = node
    total += D[cur][0]
    return total


def _kmeans_cluster(points, k, max_iters=50):
    """Simple k-means clustering. Returns list of lists of indices."""
    n = len(points)
    if n <= k:
        return [[i] for i in range(n)]

    points = np.array(points, dtype=np.float64)
    # Initialize with k-means++
    centers = [points[0].copy()]
    for _ in range(1, k):
        d2 = np.array([min(np.sum((points[i] - c) ** 2) for c in centers) for i in range(n)])
        total = d2.sum()
        if total < 1e-10:
            centers.append(points[np.random.randint(n)].copy())
        else:
            probs = d2 / total
            idx = np.random.choice(n, p=probs)
            centers.append(points[idx].copy())

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iters):
        # Assign
        d2 = np.array([[np.sum((points[i] - centers[j]) ** 2) for j in range(k)] for i in range(n)])
        new_labels = np.argmin(d2, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # Update centers
        for j in range(k):
            mask = labels == j
            if mask.sum() > 0:
                centers[j] = points[mask].mean(axis=0)

    clusters = [[] for _ in range(k)]
    for i in range(n):
        clusters[labels[i]].append(i)
    return [c for c in clusters if c]


def _build_route_for_cluster(cluster_nodes, D, demands, ready, due, service,
                              battery_max, speed, recharge_rate, horizon, stations_set,
                              n_customers, capacity):
    """Build a route for a cluster using nearest-neighbor with battery awareness."""
    if not cluster_nodes:
        return [], 0.0

    return _nearest_neighbor_route(cluster_nodes, D, demands, ready, due, service,
                                   battery_max, speed, recharge_rate, horizon,
                                   stations_set, n_customers, capacity)


def _nearest_neighbor_route(nodes, D, demands, ready, due, service,
                            battery_max, speed, recharge_rate, horizon,
                            stations_set, n_customers, capacity):
    """Nearest neighbor construction with battery awareness."""
    remaining = list(nodes)
    route = []
    cur = 0
    t = 0.0
    battery = battery_max
    total_demand = 0

    while remaining:
        best = None
        best_score = float('inf')
        for cand in remaining:
            if 1 <= cand <= n_customers:
                if total_demand + demands[cand] > capacity:
                    continue
            d = D[cur][cand]
            new_battery = battery - d
            if new_battery < -1e-9:
                continue
            new_t = t + d / speed
            if cand in stations_set:
                new_t += (battery_max - new_battery) * recharge_rate
            else:
                new_t = max(new_t, ready[cand])
                if new_t > due[cand] + 1e-9:
                    continue
            # Score: prefer closer nodes, but also consider time window slack
            slack = due[cand] - new_t if cand not in stations_set else 0
            score = d - 0.1 * min(slack, 50)  # slight preference for nodes with more slack
            if score < best_score:
                best_score = score
                best = cand

        if best is None:
            # Try any remaining node that's reachable
            for cand in remaining:
                d = D[cur][cand]
                if battery - d >= -1e-9:
                    best = cand
                    break
            if best is None:
                best = remaining[0]

        d = D[cur][best]
        battery -= d
        t += d / speed
        cur = best
        route.append(best)
        remaining.remove(best)

        if best in stations_set:
            t += (battery_max - battery) * recharge_rate
            battery = battery_max
        else:
            t = max(t, ready[best])
            t += service[best]
            total_demand += demands[best]

    return route, _compute_distance(route, D)


def _build_route_bounded(cluster_nodes, D, demands, ready, due, service,
                         battery_max, speed, recharge_rate, horizon, stations_set,
                         n_customers, capacity, deadline=None):
    """Build route with 2-opt improvement."""
    route, d = _nearest_neighbor_route(cluster_nodes, D, demands, ready, due, service,
                                       battery_max, speed, recharge_rate, horizon,
                                       stations_set, n_customers, capacity)
    # Apply 2-opt
    route = _two_opt(route, D, demands, ready, due, service,
                     capacity, battery_max, speed, recharge_rate, horizon,
                     stations_set, n_customers, max_iters=15, deadline=deadline)
    return route, _compute_distance(route, D)


def _two_opt(route, D, demands, ready, due, service,
             capacity, battery_max, speed, recharge_rate, horizon,
             stations_set, n_customers, max_iters=20, deadline=None):
    """2-opt improvement with feasibility check."""
    if len(route) < 2:
        return route

    best_route = list(route)
    best_dist = _compute_distance(best_route, D)
    improved = True
    iters = 0

    while improved and iters < max_iters:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        iters += 1
        n = len(best_route)
        for i in range(n - 1):
            for j in range(i + 1, n):
                candidate = best_route[:i] + best_route[i:j + 1][::-1] + best_route[j + 1:]
                valid, d = _validate_route(candidate, D, demands, ready, due, service,
                                           capacity, battery_max, speed, recharge_rate,
                                           horizon, stations_set, n_customers)
                if valid and d < best_dist - 1e-9:
                    best_route = candidate
                    best_dist = d
                    improved = True

    return best_route


def _or_opt(route, D, demands, ready, due, service,
            capacity, battery_max, speed, recharge_rate, horizon,
            stations_set, n_customers, max_iters=10, deadline=None):
    """Or-opt: move 1-3 consecutive nodes to a different position."""
    if len(route) < 4:
        return route

    best_route = list(route)
    best_dist = _compute_distance(best_route, D)
    improved = True
    iters = 0

    while improved and iters < max_iters:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        iters += 1
        n = len(best_route)
        for seg_len in range(1, 4):
            for i in range(n - seg_len + 1):
                segment = best_route[i:i + seg_len]
                rest = best_route[:i] + best_route[i + seg_len:]
                for j in range(len(rest) + 1):
                    if i > 0 and j <= i:
                        continue
                    if i + seg_len <= j:
                        continue
                    candidate = rest[:j] + segment + rest[j:]
                    valid, d = _validate_route(candidate, D, demands, ready, due, service,
                                               capacity, battery_max, speed, recharge_rate,
                                               horizon, stations_set, n_customers)
                    if valid and d < best_dist - 1e-9:
                        best_route = candidate
                        best_dist = d
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    return best_route


def _try_insert_station(route, D, demands, ready, due, service,
                        capacity, battery_max, speed, recharge_rate, horizon,
                        stations_set, n_customers):
    """Try to insert a station at the best position in the route if it helps."""
    best_route = list(route)
    best_dist = _compute_distance(best_route, D)

    for station in stations_set:
        # Try inserting at each position
        for pos in range(len(best_route) + 1):
            candidate = best_route[:pos] + [station] + best_route[pos:]
            valid, d = _validate_route(candidate, D, demands, ready, due, service,
                                       capacity, battery_max, speed, recharge_rate,
                                       horizon, stations_set, n_customers)
            if valid and d < best_dist - 1e-9:
                best_route = candidate
                best_dist = d

    return best_route


def _inter_route_merge(routes, D, demands, ready, due, service,
                       capacity, battery_max, speed, recharge_rate, horizon,
                       stations_set, n_customers, deadline=None):
    """Try to merge pairs of routes if feasible."""
    if len(routes) < 2:
        return routes

    merged = True
    while merged:
        if deadline is not None and time.time() > deadline:
            break
        merged = False
        best_merge = None
        best_saving = 0

        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                # Try concatenation in both orders
                for combined in [routes[i] + routes[j], routes[j] + routes[i]]:
                    valid, d = _validate_route(combined, D, demands, ready, due, service,
                                               capacity, battery_max, speed, recharge_rate,
                                               horizon, stations_set, n_customers)
                    if valid:
                        old_d = _compute_distance(routes[i], D) + _compute_distance(routes[j], D)
                        saving = old_d - d
                        if saving > best_saving + 1e-9:
                            best_saving = saving
                            best_merge = (i, j, combined)

        if best_merge:
            i, j, combined = best_merge
            routes.pop(j)
            routes.pop(i)
            routes.append(combined)
            merged = True

    return routes


def _inter_route_move(routes, D, demands, ready, due, service,
                      capacity, battery_max, speed, recharge_rate, horizon,
                      stations_set, n_customers, max_moves=50, deadline=None):
    """Try moving a customer from one route to another."""
    improved = True
    moves = 0
    while improved and moves < max_moves:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        moves += 1
        for i in range(len(routes)):
            for j in range(len(routes[i])):
                if 1 <= routes[i][j] <= n_customers:
                    cust = routes[i][j]
                    for k in range(len(routes)):
                        if k == i:
                            continue
                        # Try inserting cust at each position in route k
                        for pos in range(len(routes[k]) + 1):
                            new_i = routes[i][:j] + routes[i][j + 1:]
                            new_k = routes[k][:pos] + [cust] + routes[k][pos:]

                            # Check capacity
                            demand_i = sum(demands[v] for v in new_i if 1 <= v <= n_customers)
                            demand_k = sum(demands[v] for v in new_k if 1 <= v <= n_customers)
                            if demand_k > capacity:
                                continue

                            valid_i, d_i = _validate_route(new_i, D, demands, ready, due, service,
                                                           capacity, battery_max, speed, recharge_rate,
                                                           horizon, stations_set, n_customers) if new_i else (True, 0.0)
                            valid_k, d_k = _validate_route(new_k, D, demands, ready, due, service,
                                                           capacity, battery_max, speed, recharge_rate,
                                                           horizon, stations_set, n_customers)

                            if valid_i and valid_k:
                                old_d = _compute_distance(routes[i], D) + _compute_distance(routes[k], D)
                                new_d = d_i + d_k
                                if new_d < old_d - 1e-9:
                                    routes[i] = new_i
                                    routes[k] = new_k
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
            if improved:
                break

    # Remove empty routes
    routes = [r for r in routes if r]
    return routes


def _savings_construct(customers, D, demands, ready, due, service,
                       capacity, battery_max, speed, recharge_rate, horizon,
                       stations_set, n_customers, deadline=None):
    """Savings-based construction with auto-station insertion."""
    # Start with each customer as a separate route
    routes = {c: [c] for c in customers}
    # Compute savings: savings(c,d) = dist(c,0) + dist(0,d) - dist(c,d)
    savings_list = []
    for i in customers:
        for j in customers:
            if i != j:
                s = D[i][0] + D[0][j] - D[i][j]
                savings_list.append((s, i, j))
    savings_list.sort(reverse=True)

    # Merge routes based on savings
    for s, i, j in savings_list:
        if deadline is not None and time.time() > deadline:
            break
        if s <= 0:
            break
        # Find which routes contain i and j
        ri = None
        rj = None
        for r in routes:
            if i in routes[r]:
                ri = r
            if j in routes[r]:
                rj = r
        if ri is None or rj is None or ri == rj:
            continue

        # Try merging
        route_i = routes[ri]
        route_j = routes[rj]

        # Check capacity
        demand_i = sum(demands[c] for c in route_i if 1 <= c <= n_customers)
        demand_j = sum(demands[c] for c in route_j if 1 <= c <= n_customers)
        if demand_i + demand_j > capacity:
            continue

        # Try both orders
        best_merged = None
        best_d = float('inf')
        for combined in [route_i + route_j, route_j + route_i]:
            valid, d = _validate_route(combined, D, demands, ready, due, service,
                                       capacity, battery_max, speed, recharge_rate,
                                       horizon, stations_set, n_customers)
            if valid and d < best_d:
                best_d = d
                best_merged = combined

        if best_merged is not None:
            old_d = _compute_distance(route_i, D) + _compute_distance(route_j, D)
            if best_d < old_d - 1e-9:
                del routes[ri]
                del routes[rj]
                routes[ri] = best_merged

    return list(routes.values())


def _auto_insert_stations(route, D, demands, ready, due, service,
                          capacity, battery_max, speed, recharge_rate, horizon,
                          stations_set, n_customers, max_inserts=10):
    """Automatically insert stations where needed to make route feasible."""
    for _ in range(max_inserts):
        valid, d = _validate_route(route, D, demands, ready, due, service,
                                   capacity, battery_max, speed, recharge_rate,
                                   horizon, stations_set, n_customers)
        if valid:
            break

        # Find where battery goes negative or time window is violated
        cur = 0
        t = 0.0
        battery = battery_max
        insert_pos = None
        insert_station = None

        for idx, node in enumerate(route):
            d_seg = D[cur][node]
            new_battery = battery - d_seg
            new_t = t + d_seg / speed

            if new_battery < -1e-9:
                # Need to charge before this node
                # Find best station to insert before this position
                for st in stations_set:
                    d_to_st = D[cur][st]
                    d_st_to_node = D[st][node]
                    if battery - d_to_st >= -1e-9:
                        # Check if we can reach station
                        st_battery = battery - d_to_st
                        st_time = t + d_to_st / speed
                        st_time += (battery_max - st_battery) * recharge_rate
                        st_battery = battery_max
                        new_t2 = st_time + d_st_to_node / speed
                        if node in stations_set:
                            new_t2 += (battery_max - (st_battery - d_st_to_node)) * recharge_rate
                        else:
                            new_t2 = max(new_t2, ready[node])
                            if new_t2 > due[node] + 1e-9:
                                continue
                        # This station could work
                        if insert_pos is None or d_to_st + d_st_to_node < D[cur][route[insert_pos]]:
                            insert_pos = idx
                            insert_station = st
                        break
            else:
                # Check time window
                if node not in stations_set:
                    wait_t = max(new_t, ready[node])
                    if wait_t > due[node] + 1e-9:
                        # Time window violation - might need to delay or reorder
                        # For now, try inserting a station earlier to adjust timing
                        pass

            # Update state
            if node in stations_set:
                t = new_t + (battery_max - new_battery) * recharge_rate
                battery = battery_max
            else:
                t = max(new_t, ready[node]) + service[node]
                battery = new_battery
            cur = node

        if insert_pos is not None and insert_station is not None:
            route = route[:insert_pos] + [insert_station] + route[insert_pos:]
        else:
            break

    return route


def plan(instance):
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    coords = np.array(instance["coords"], dtype=np.float64)
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations_set = set(instance["stations"])
    n_customers = instance["n_customers"]

    # Build distance matrix
    D = _build_dist_matrix(coords)

    customers = list(range(1, n_customers + 1))

    # Try savings-based construction first
    if time.time() < deadline - 60:
        routes = _savings_construct(customers, D, demands, ready, due, service,
                                    capacity, battery_max, speed, recharge_rate, horizon,
                                    stations_set, n_customers, deadline=deadline - 30)
    else:
        # Fallback: cluster-based
        cust_coords = [coords[c] for c in customers]
        total_demand = sum(demands[c] for c in customers)
        n_clusters = max(2, int(math.ceil(total_demand / capacity)))
        n_clusters = min(n_clusters, len(customers))
        clusters_idx = _kmeans_cluster(cust_coords, n_clusters)
        clusters = [[customers[i] for i in c] for c in clusters_idx]
        final_clusters = []
        for cl in clusters:
            cl_demand = sum(demands[c] for c in cl)
            if cl_demand <= capacity:
                final_clusters.append(cl)
            else:
                for c in cl:
                    final_clusters.append([c])
        routes = []
        for cl in final_clusters:
            if time.time() > deadline - 10:
                break
            route, d = _build_route_for_cluster(cl, D, demands, ready, due, service,
                                                battery_max, speed, recharge_rate, horizon,
                                                stations_set, n_customers, capacity)
            if route:
                routes.append(route)

    # Check all customers are covered
    covered = set()
    for r in routes:
        for node in r:
            if 1 <= node <= n_customers:
                covered.add(node)
    missing = [c for c in customers if c not in covered]
    for c in missing:
        routes.append([c])

    # Auto-insert stations to make routes feasible
    for i in range(len(routes)):
        if time.time() > deadline - 20:
            break
        routes[i] = _auto_insert_stations(routes[i], D, demands, ready, due, service,
                                          capacity, battery_max, speed, recharge_rate,
                                          horizon, stations_set, n_customers)

    # Validate
    valid, total_dist = _validate_solution(routes, D, demands, ready, due, service,
                                           capacity, battery_max, speed, recharge_rate,
                                           horizon, stations_set, n_customers)
    if not valid:
        # Fallback: one customer per route
        return [[v] for v in customers]

    # Local optimization: 2-opt + or-opt on each route
    for i in range(len(routes)):
        if time.time() > deadline - 15:
            break
        routes[i] = _two_opt(routes[i], D, demands, ready, due, service,
                             capacity, battery_max, speed, recharge_rate, horizon,
                             stations_set, n_customers, max_iters=12, deadline=deadline - 10)
        routes[i] = _or_opt(routes[i], D, demands, ready, due, service,
                            capacity, battery_max, speed, recharge_rate, horizon,
                            stations_set, n_customers, max_iters=8, deadline=deadline - 10)

    # Inter-route: try merging
    if time.time() < deadline - 10:
        routes = _inter_route_merge(routes, D, demands, ready, due, service,
                                    capacity, battery_max, speed, recharge_rate,
                                    horizon, stations_set, n_customers, deadline=deadline - 5)

    # Inter-route: try moving customers
    if time.time() < deadline - 5:
        routes = _inter_route_move(routes, D, demands, ready, due, service,
                                   capacity, battery_max, speed, recharge_rate, horizon,
                                   stations_set, n_customers, max_moves=30, deadline=deadline - 2)

    # Final 2-opt pass
    for i in range(len(routes)):
        if time.time() > deadline - 2:
            break
        routes[i] = _two_opt(routes[i], D, demands, ready, due, service,
                             capacity, battery_max, speed, recharge_rate, horizon,
                             stations_set, n_customers, max_iters=8, deadline=deadline)

    # Final validation
    valid, total_dist = _validate_solution(routes, D, demands, ready, due, service,
                                           capacity, battery_max, speed, recharge_rate,
                                           horizon, stations_set, n_customers)
    if not valid:
        return [[v] for v in customers]

    return routes
# EVOLVE-BLOCK-END
