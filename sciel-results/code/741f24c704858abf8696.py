"""Combined program: spatial clustering initial solution + Or-opt/relocate/2-opt local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import sqrt

def _get_distance_matrix(coords):
    """Precompute Euclidean distance matrix using NumPy."""
    arr = np.array(coords, dtype=np.float64)
    dx = arr[:, 0, None] - arr[None, :, 0]
    dy = arr[:, 1, None] - arr[None, :, 1]
    return np.sqrt(dx**2 + dy**2)

def _route_distance_fast(dist_matrix, route):
    """Calculate total distance of a route using precomputed distance matrix."""
    if not route:
        return 0.0
    d = dist_matrix[0][route[0]]
    for k in range(len(route) - 1):
        d += dist_matrix[route[k]][route[k + 1]]
    d += dist_matrix[route[-1]][0]
    return d

def _check_route_feasibility_fast(dist_matrix, demands, ready, due, service,
                                  battery, recharge_rate, horizon, route,
                                  capacity, n_customers, stations):
    """
    Check if a single route is feasible using precomputed distances.
    Returns (feasible, total_distance).
    """
    if not route:
        return True, 0.0

    total_demand = 0
    for v in route:
        if v <= n_customers:
            total_demand += demands[v]
            if total_demand > capacity:
                return False, 0.0

    t = 0.0
    b = battery
    prev = 0
    total_dist = 0.0

    for v in route:
        d = dist_matrix[prev][v]
        t += d
        b -= d
        total_dist += d
        if b < -1e-9:
            return False, 0.0

        if v <= n_customers:
            if t < ready[v]:
                t = ready[v]
            if t > due[v] + 1e-9:
                return False, 0.0
            t += service[v]
        else:
            time_charging = (battery - b) * recharge_rate
            t += time_charging
            b = battery

        prev = v

    d = dist_matrix[prev][0]
    t += d
    b -= d
    total_dist += d
    if b < -1e-9:
        return False, 0.0
    if t > horizon + 1e-9:
        return False, 0.0

    return True, total_dist

def _cluster_customers(dist_matrix, demands, all_customers, n_clusters, capacity,
                       n_customers, battery, seed=0):
    """
    K-means-style spatial clustering of customers into n_clusters groups.
    Returns list of lists of customer ids.
    """
    rng = np.random.RandomState(seed)
    n = len(all_customers)
    if n <= n_clusters:
        return [[c] for c in all_customers]

    # Initial centroids: pick n_clusters customers spread out
    # Use farthest-point sampling from depot for good spread
    centroids = [all_customers[0]]
    min_dist_to_centroid = np.array([dist_matrix[0][c] for c in all_customers], dtype=np.float64)
    # Mark first
    for _ in range(n_clusters - 1):
        # Pick customer farthest from current centroids
        idx = int(np.argmax(min_dist_to_centroid))
        centroids.append(all_customers[idx])
        d_to_new = np.array([dist_matrix[c][all_customers[idx]] for c in all_customers], dtype=np.float64)
        min_dist_to_centroid = np.minimum(min_dist_to_centroid, d_to_new)

    # K-means iterations
    assign = np.zeros(n, dtype=int)
    for it in range(20):
        # Assign each customer to nearest centroid
        for i, c in enumerate(all_customers):
            best = 0
            best_d = float('inf')
            for k, cen in enumerate(centroids):
                d = dist_matrix[c][cen]
                if d < best_d:
                    best_d = d
                    best = k
            assign[i] = best

        # Update centroids (mean position)
        new_centroids = []
        for k in range(n_clusters):
            members = [all_customers[i] for i in range(n) if assign[i] == k]
            if members:
                xs = np.mean([dist_matrix[0][0] for _ in members])  # placeholder
                # Actually compute mean of coords via dist? We need coords. Use representative: closest to centroid.
                # We don't have coords here; use the centroid as the member closest to old centroid.
                best_m = members[0]
                best_d = float('inf')
                for m in members:
                    d = dist_matrix[m][centroids[k]]
                    if d < best_d:
                        best_d = d
                        best_m = m
                new_centroids.append(best_m)
            else:
                new_centroids.append(centroids[k])
        if new_centroids == centroids:
            break
        centroids = new_centroids

    # Build clusters
    clusters = [[] for _ in range(n_clusters)]
    for i, c in enumerate(all_customers):
        clusters[assign[i]].append(c)

    # Split any cluster that exceeds capacity
    final_clusters = []
    for cl in clusters:
        if not cl:
            continue
        total = sum(demands[c] for c in cl)
        if total <= capacity:
            final_clusters.append(cl)
        else:
            # Split into sub-clusters by capacity
            cur = []
            cur_d = 0
            for c in cl:
                if cur_d + demands[c] > capacity and cur:
                    final_clusters.append(cur)
                    cur = []
                    cur_d = 0
                cur.append(c)
                cur_d += demands[c]
            if cur:
                final_clusters.append(cur)
    return final_clusters

def _nearest_neighbor_route(dist_matrix, customers, ready, due, service,
                            battery, recharge_rate, horizon, capacity,
                            n_customers, stations):
    """Build a feasible route from a set of customers via nearest neighbor + feasibility check."""
    if not customers:
        return []
    route = []
    remaining = list(customers)
    # Sort by due time as tie-breaker
    remaining.sort(key=lambda c: due[c])
    prev = 0
    t = 0.0
    b = battery
    total_demand = 0
    while remaining:
        best_c = None
        best_d = float('inf')
        for c in remaining:
            d = dist_matrix[prev][c]
            if d < best_d:
                best_d = d
                best_c = c
        # Try adding best_c
        cand = route + [best_c]
        ok, _ = _check_route_feasibility_fast(
            dist_matrix, [0]*(n_customers+1), ready, due, service,
            battery, recharge_rate, horizon, cand,
            capacity, n_customers, stations
        )
        # Need real demands; pass through properly
        ok, _ = _check_route_feasibility_fast(
            dist_matrix, _DEMANDS, ready, due, service,
            battery, recharge_rate, horizon, cand,
            capacity, n_customers, stations
        )
        if ok and total_demand + _DEMANDS[best_c] <= capacity:
            route.append(best_c)
            total_demand += _DEMANDS[best_c]
            remaining.remove(best_c)
            # Update t, b by simulating
            d = dist_matrix[prev][best_c]
            t += d
            b -= d
            if best_c <= n_customers:
                if t < ready[best_c]:
                    t = ready[best_c]
                t += service[best_c]
            prev = best_c
        else:
            # Cannot add best_c feasibly; try next nearest
            # Find first feasible customer
            found = False
            for c in remaining:
                cand = route + [c]
                ok, _ = _check_route_feasibility_fast(
                    dist_matrix, _DEMANDS, ready, due, service,
                    battery, recharge_rate, horizon, cand,
                    capacity, n_customers, stations
                )
                if ok and total_demand + _DEMANDS[c] <= capacity:
                    route.append(c)
                    total_demand += _DEMANDS[c]
                    remaining.remove(c)
                    d = dist_matrix[prev][c]
                    t += d
                    b -= d
                    if c <= n_customers:
                        if t < ready[c]:
                            t = ready[c]
                        t += service[c]
                    prev = c
                    found = True
                    break
            if not found:
                break
    return route

# Global placeholder for demands used in _nearest_neighbor_route
_DEMANDS = None

def _solve_instance(instance):
    global _DEMANDS
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance.get("speed", 1.0)
    recharge_rate = instance.get("recharge_rate", 0.5)
    horizon = instance["horizon"]
    _DEMANDS = demands

    dist_matrix = _get_distance_matrix(coords)
    all_customers = list(range(1, n_customers + 1))

    # Determine number of clusters
    total_demand = sum(demands[c] for c in all_customers)
    k_capacity = max(1, int(np.ceil(total_demand / capacity)))
    # Battery heuristic: max distance a vehicle can travel round trip
    max_range = battery * 0.8  # leave margin
    # Estimate avg customer distance from depot
    avg_dep_dist = np.mean([dist_matrix[0][c] for c in all_customers])
    k_battery = max(1, int(np.ceil((2 * avg_dep_dist * n_customers) / max_range))) if max_range > 0 else k_capacity
    n_clusters = max(k_capacity, min(k_battery, n_customers))
    n_clusters = max(1, min(n_clusters, n_customers))

    # Build initial routes via clustering
    clusters = _cluster_customers(dist_matrix, demands, all_customers, n_clusters,
                                  capacity, n_customers, battery, seed=42)

    routes = []
    unassigned = list(all_customers)
    for cl in clusters:
        if not cl:
            continue
        route = _nearest_neighbor_route(
            dist_matrix, cl, ready, due, service,
            battery, recharge_rate, horizon, capacity,
            n_customers, stations
        )
        if route:
            routes.append(route)
            for c in route:
                if c in unassigned:
                    unassigned.remove(c)

    # Insert any unassigned customers greedily
    for c in unassigned:
        best_route_idx = None
        best_dist_increase = float('inf')
        best_new_route = None
        for i, route in enumerate(routes):
            if sum(demands[v] for v in route) + demands[c] > capacity:
                continue
            best_pos = None
            best_rd = float('inf')
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [c] + route[pos:]
                feasible, nd = _check_route_feasibility_fast(
                    dist_matrix, demands, ready, due, service,
                    battery, recharge_rate, horizon, new_route,
                    capacity, n_customers, stations
                )
                if feasible and nd < best_rd:
                    best_rd = nd
                    best_pos = pos
            if best_pos is not None:
                old_rd = _route_distance_fast(dist_matrix, route)
                increase = best_rd - old_rd
                if increase < best_dist_increase:
                    best_dist_increase = increase
                    best_route_idx = i
                    best_new_route = route[:best_pos] + [c] + route[best_pos:]
        if best_route_idx is not None:
            routes[best_route_idx] = best_new_route
        else:
            routes.append([c])

    routes = [r for r in routes if r]
    if not routes:
        return []

    # Local search with deadline
    start = time.time()
    deadline = start + 110

    def _time_left():
        return time.time() < deadline

    def _two_opt_route(route):
        if len(route) < 2:
            return route
        n = len(route)
        best_route = list(route)
        current_dist = _route_distance_fast(dist_matrix, best_route)
        improved = True
        max_iter = 30
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            if not _time_left():
                break
            for i in range(n - 1):
                if not _time_left():
                    break
                for j in range(i + 1, n):
                    if not _time_left():
                        break
                    if i == 0:
                        old1 = dist_matrix[0][best_route[i]]
                        new1 = dist_matrix[0][best_route[j]]
                    else:
                        old1 = dist_matrix[best_route[i-1]][best_route[i]]
                        new1 = dist_matrix[best_route[i-1]][best_route[j]]
                    if j == n - 1:
                        old2 = dist_matrix[best_route[j]][0]
                        new2 = dist_matrix[best_route[i]][0]
                    else:
                        old2 = dist_matrix[best_route[j]][best_route[j+1]]
                        new2 = dist_matrix[best_route[i]][best_route[j+1]]
                    delta_d = (new1 + new2) - (old1 + old2)
                    if delta_d >= -1e-6:
                        continue
                    new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                    feasible, nd = _check_route_feasibility_fast(
                        dist_matrix, demands, ready, due, service,
                        battery, recharge_rate, horizon, new_route,
                        capacity, n_customers, stations
                    )
                    if feasible and nd < current_dist - 1e-6:
                        best_route = new_route
                        current_dist = nd
                        improved = True
        return best_route

    def _or_opt_route(route):
        """Move 1-3 consecutive customers to a new position within the same route."""
        if len(route) < 3:
            return route
        n = len(route)
        best_route = list(route)
        current_dist = _route_distance_fast(dist_matrix, best_route)
        improved = True
        max_iter = 20
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            if not _time_left():
                break
            for size in (1, 2, 3):
                if not _time_left():
                    break
                for i in range(n - size + 1):
                    if not _time_left():
                        break
                    segment = best_route[i:i+size]
                    rest = best_route[:i] + best_route[i+size:]
                    for j in range(len(rest) + 1):
                        if j == i or (i > 0 and j == i + size - 1):
                            # Skip inserting back at same spot
                            continue
                        new_route = rest[:j] + segment + rest[j:]
                        if new_route == best_route:
                            continue
                        feasible, nd = _check_route_feasibility_fast(
                            dist_matrix, demands, ready, due, service,
                            battery, recharge_rate, horizon, new_route,
                            capacity, n_customers, stations
                        )
                        if feasible and nd < current_dist - 1e-6:
                            best_route = new_route
                            current_dist = nd
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return best_route

    # Inter-route relocate: move one customer between routes
    def _relocate_pass():
        nonlocal routes
        if not _time_left():
            return False
        improved = False
        for c in all_customers:
            if not _time_left():
                break
            curr_idx = -1
            for i, r in enumerate(routes):
                if c in r:
                    curr_idx = i
                    break
            if curr_idx == -1:
                continue
            curr_route = routes[curr_idx]
            remaining_curr = [v for v in curr_route if v != c]
            dist_curr_without = _route_distance_fast(dist_matrix, remaining_curr) if remaining_curr else 0.0
            best_target = None
            best_delta = 0.0
            best_state = None
            for i, route in enumerate(routes):
                if not _time_left():
                    break
                if i == curr_idx:
                    target_route = remaining_curr
                    old_target_dist = dist_curr_without
                else:
                    target_route = route
                    old_target_dist = _route_distance_fast(dist_matrix, route)
                if sum(demands[v] for v in target_route) + demands[c] > capacity:
                    continue
                best_pos = None
                best_td = float('inf')
                for pos in range(len(target_route) + 1):
                    new_target = target_route[:pos] + [c] + target_route[pos:]
                    feasible, nd = _check_route_feasibility_fast(
                        dist_matrix, demands, ready, due, service,
                        battery, recharge_rate, horizon, new_target,
                        capacity, n_customers, stations
                    )
                    if feasible and nd < best_td:
                        best_td = nd
                        best_pos = pos
                if best_pos is not None:
                    if i == curr_idx:
                        old_affected = _route_distance_fast(dist_matrix, curr_route)
                        new_affected = best_td
                        delta = new_affected - old_affected
                    else:
                        old_affected = _route_distance_fast(dist_matrix, curr_route) + old_target_dist
                        new_affected = dist_curr_without + best_td
                        delta = new_affected - old_affected
                    if delta < best_delta - 1e-6:
                        best_delta = delta
                        best_target = i
                        new_routes = [list(r) for r in routes]
                        if i == curr_idx:
                            new_routes[curr_idx] = target_route[:best_pos] + [c] + target_route[best_pos:]
                        else:
                            new_routes[curr_idx] = remaining_curr
                            new_routes[i] = route[:best_pos] + [c] + route[best_pos:]
                        best_state = new_routes
            if best_target is not None:
                routes = best_state
                improved = True
        return improved

    # Main local search loop
    max_rounds = 30
    for rnd in range(max_rounds):
        if not _time_left():
            break
        # Intra-route 2-opt
        for i in range(len(routes)):
            if not _time_left():
                break
            new_r = _two_opt_route(routes[i])
            if new_r != routes[i]:
                routes[i] = new_r
        # Intra-route Or-opt
        for i in range(len(routes)):
            if not _time_left():
                break
            new_r = _or_opt_route(routes[i])
            if new_r != routes[i]:
                routes[i] = new_r
        # Inter-route relocate
        _relocate_pass()
        # Remove empty routes
        routes = [r for r in routes if r]
        if not routes:
            break

    # Final 2-opt pass
    for i in range(len(routes)):
        routes[i] = _two_opt_route(routes[i])
    routes = [r for r in routes if r]

    return routes

def plan(instance):
    start_time = time.time()
    deadline = start_time + 110

    routes = _solve_instance(instance)

    # Final verification
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery = instance["battery"]
    recharge_rate = instance.get("recharge_rate", 0.5)
    horizon = instance["horizon"]

    dist_matrix = _get_distance_matrix(coords)

    all_ok = True
    served = set()
    for r in routes:
        ok, _ = _check_route_feasibility_fast(
            dist_matrix, demands, ready, due, service,
            battery, recharge_rate, horizon, r,
            capacity, n_customers, stations
        )
        if not ok:
            all_ok = False
            break
        for v in r:
            if v <= n_customers:
                if v in served:
                    all_ok = False
                    break
                served.add(v)
        if not all_ok:
            break
    if len(served) != n_customers:
        all_ok = False

    if not all_ok:
        customers = list(range(1, n_customers + 1))
        routes = [[c] for c in customers]

    return routes
# EVOLVE-BLOCK-END
