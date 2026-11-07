"""EV routing with time windows and battery constraints - improved approach."""
import numpy as np
import time
import random
import math
from collections import defaultdict

DEADLINE_SECONDS = 145.0


def _build_dist_matrix(coords):
    coords = np.array(coords, dtype=np.float64)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _simulate_route(route, dist_matrix, instance, n_customers, stations_set,
                    battery_max, speed, recharge_rate, horizon, capacity):
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]

    total_demand = 0
    for node in route:
        if 1 <= node <= n_customers:
            total_demand += demands[node]
    if total_demand > capacity:
        return False, 0.0, 0.0, 0.0

    battery = battery_max
    time_at = 0.0
    distance = 0.0
    current_node = 0
    inv_speed = 1.0 / speed

    for node in route:
        d = dist_matrix[current_node, node]
        time_at += d * inv_speed
        battery -= d
        distance += d

        if battery < -1e-9:
            return False, 0.0, 0.0, 0.0

        if 1 <= node <= n_customers:
            r = ready[node]
            if time_at < r:
                time_at = r
            if time_at > due[node] + 1e-9:
                return False, 0.0, 0.0, 0.0
            time_at += service[node]
        elif node in stations_set:
            energy_needed = battery_max - battery
            if energy_needed > 0:
                time_at += energy_needed * recharge_rate
            battery = battery_max

        current_node = node

    d = dist_matrix[current_node, 0]
    time_at += d * inv_speed
    battery -= d
    distance += d

    if battery < -1e-9:
        return False, 0.0, 0.0, 0.0
    if time_at > horizon + 1e-9:
        return False, 0.0, 0.0, 0.0

    return True, distance, time_at, battery


def _validate_solution(routes, instance):
    n_customers = instance["n_customers"]
    coords = instance["coords"]
    dist_matrix = _build_dist_matrix(coords)
    stations_set = set(instance["stations"])
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    capacity = instance["capacity"]

    served = set()
    total_distance = 0.0

    for route in routes:
        valid, dist, _, _ = _simulate_route(route, dist_matrix, instance, n_customers,
                                             stations_set, battery_max, speed,
                                             recharge_rate, horizon, capacity)
        if not valid:
            return False, 0.0
        total_distance += dist
        for node in route:
            if 1 <= node <= n_customers:
                if node in served:
                    return False, 0.0
                served.add(node)

    if served != set(range(1, n_customers + 1)):
        return False, 0.0

    return True, total_distance


def _compute_total_distance(routes, dist_matrix):
    total = 0.0
    for route in routes:
        current = 0
        for node in route:
            total += dist_matrix[current, node]
            current = node
        total += dist_matrix[current, 0]
    return total


def _route_feasible(route, dist_matrix, instance, n_customers, stations_set,
                    battery_max, speed, recharge_rate, horizon, capacity):
    valid, _, _, _ = _simulate_route(route, dist_matrix, instance, n_customers,
                                      stations_set, battery_max, speed,
                                      recharge_rate, horizon, capacity)
    return valid


def _solve_tsp_dp(points, dist_matrix, max_points=12):
    """Solve TSP on a small set of points using DP (Held-Karp).
    Returns optimal tour as list of point indices."""
    n = len(points)
    if n <= 1:
        return list(range(n))
    if n > max_points:
        return list(range(n))

    full_mask = (1 << n) - 1
    INF = float('inf')
    dp = [[INF] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]
    dp[1][0] = 0

    for mask in range(1 << n):
        if not (mask & 1):
            continue
        for u in range(n):
            if dp[mask][u] == INF:
                continue
            if not (mask & (1 << u)):
                continue
            for v in range(n):
                if mask & (1 << v):
                    continue
                new_mask = mask | (1 << v)
                cost = dist_matrix[points[u], points[v]]
                if dp[new_mask][v] > dp[mask][u] + cost:
                    dp[new_mask][v] = dp[mask][u] + cost
                    parent[new_mask][v] = u

    best_end = -1
    best_cost = INF
    for v in range(n):
        if v == 0:
            continue
        if dp[full_mask][v] + dist_matrix[points[v], points[0]] < best_cost:
            best_cost = dp[full_mask][v] + dist_matrix[points[v], points[0]]
            best_end = v

    if best_end == -1:
        return list(range(n))

    tour = []
    mask = full_mask
    cur = best_end
    while cur != -1:
        tour.append(cur)
        prev = parent[mask][cur]
        mask ^= (1 << cur)
        cur = prev
    tour.reverse()
    return tour


def _savings_construct(instance, dist_matrix, n_customers, stations_set,
                       battery_max, speed, recharge_rate, horizon, capacity,
                       seed=0):
    random.seed(seed)
    demands = instance["demands"]

    routes = [[i] for i in range(1, n_customers + 1)]
    route_loads = [demands[i] for i in range(1, n_customers + 1)]

    savings = []
    for i in range(1, n_customers + 1):
        for j in range(i + 1, n_customers + 1):
            s = dist_matrix[0, i] + dist_matrix[0, j] - dist_matrix[i, j]
            if s > 1e-6:
                savings.append((s, i, j))

    savings.sort(key=lambda x: x[0], reverse=True)

    cust_to_route = {i: i - 1 for i in range(1, n_customers + 1)}

    for s, i, j in savings:
        ri = cust_to_route[i]
        rj = cust_to_route[j]

        if ri == rj:
            continue

        route_i = routes[ri]
        route_j = routes[rj]

        if route_i[-1] == i and route_j[0] == j:
            new_route = route_i + route_j
        elif route_i[0] == i and route_j[-1] == j:
            new_route = route_i + route_j[::-1]
        elif route_i[-1] == i and route_j[-1] == j:
            new_route = route_i + route_j[::-1]
        elif route_i[0] == i and route_j[0] == j:
            new_route = route_i[::-1] + route_j
        else:
            continue

        if route_loads[ri] + route_loads[rj] > capacity:
            continue

        if _route_feasible(new_route, dist_matrix, instance, n_customers,
                           stations_set, battery_max, speed, recharge_rate,
                           horizon, capacity):
            routes[ri] = new_route
            routes[rj] = []
            route_loads[ri] += route_loads[rj]

            for cust in new_route:
                cust_to_route[cust] = ri

    valid_routes = [r for r in routes if r]
    return valid_routes


def _insert_station_optimal(route, dist_matrix, instance, n_customers, stations_set,
                            battery_max, speed, recharge_rate, horizon, capacity):
    """Insert a charging station at the optimal position in the route."""
    best_new_route = None
    best_cost = float('inf')
    for station in stations_set:
        for pos in range(len(route) + 1):
            new_route = route[:pos] + [station] + route[pos:]
            if _route_feasible(new_route, dist_matrix, instance, n_customers,
                                stations_set, battery_max, speed, recharge_rate,
                                horizon, capacity):
                cost = _compute_total_distance([new_route], dist_matrix)
                if cost < best_cost:
                    best_cost = cost
                    best_new_route = new_route
    if best_new_route is not None:
        return best_new_route, True
    return route, False


def _two_opt(route, dist_matrix, instance, n_customers, stations_set,
             battery_max, speed, recharge_rate, horizon, capacity, max_iters=30):
    if len(route) < 2:
        return route

    def is_feasible(r):
        return _route_feasible(r, dist_matrix, instance, n_customers, stations_set,
                                battery_max, speed, recharge_rate, horizon, capacity)

    def route_dist(r):
        d = 0.0
        prev = 0
        for node in r:
            d += dist_matrix[prev, node]
            prev = node
        d += dist_matrix[prev, 0]
        return d

    best_route = route[:]
    best_dist = route_dist(best_route)

    for _ in range(max_iters):
        improved = False
        for i in range(len(best_route) - 1):
            for j in range(i + 1, len(best_route)):
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                if is_feasible(new_route):
                    new_dist = route_dist(new_route)
                    if new_dist < best_dist - 1e-9:
                        best_route = new_route
                        best_dist = new_dist
                        improved = True
        if not improved:
            break

    return best_route


def _advanced_local_search(routes, dist_matrix, instance, n_customers, stations_set,
                           battery_max, speed, recharge_rate, horizon, capacity, deadline):
    demands = instance["demands"]

    def is_feasible(route):
        return _route_feasible(route, dist_matrix, instance, n_customers, stations_set,
                                battery_max, speed, recharge_rate, horizon, capacity)

    def route_load(route):
        return sum(demands[node] for node in route if 1 <= node <= n_customers)

    def get_total_dist(rts):
        return _compute_total_distance(rts, dist_matrix)

    current_dist = get_total_dist(routes)
    n_routes = len(routes)
    
    # Precompute route distances
    route_dists = [0.0] * n_routes
    for i in range(n_routes):
        route_dists[i] = _compute_total_distance([routes[i]], dist_matrix)

    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for r_idx in range(n_routes):
            if time.time() > deadline - 5:
                break
            route = routes[r_idx]
            n = len(route)
            if n < 2:
                continue

            # 2-Opt with incremental distance calculation
            # Note: 2-opt reverses a segment. The change in distance is:
            # - dist(prev, start) - dist(end, next) + dist(prev, end) + dist(start, next)
            # where prev is node before start, next is node after end.
            # If start is 0, prev is depot (0). If end is n-1, next is depot (0).
            
            best_improvement = 0.0
            best_move = None
            
            for i in range(n - 1):
                for j in range(i + 1, n):
                    # Nodes involved in the cut
                    # Segment is route[i:j+1]
                    # Prev node is route[i-1] if i>0 else 0
                    # Next node is route[j+1] if j<n-1 else 0
                    prev_node = route[i-1] if i > 0 else 0
                    next_node = route[j+1] if j < n - 1 else 0
                    start_node = route[i]
                    end_node = route[j]
                    
                    # Old connections
                    old_cost = dist_matrix[prev_node, start_node] + dist_matrix[end_node, next_node]
                    # New connections
                    new_cost = dist_matrix[prev_node, end_node] + dist_matrix[start_node, next_node]
                    
                    delta = new_cost - old_cost
                    if delta < best_improvement - 1e-9:
                        best_improvement = delta
                        best_move = (i, j)

            if best_move is not None:
                i, j = best_move
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                if is_feasible(new_route):
                    routes[r_idx] = new_route
                    route_dists[r_idx] += best_improvement
                    current_dist += best_improvement
                    improved = True
                    continue # Restart loop for this route or next? Usually next is fine, or restart. Let's do next to keep moving.

            # Or-Opt (move segment of size 1, 2, 3)
            # This is more complex to calculate incrementally without full simulation for feasibility,
            # but we can estimate distance change.
            for k in [1, 2, 3]:
                if n <= k:
                    continue
                best_oropt_improvement = 0.0
                best_oropt_move = None
                
                for i in range(n - k + 1):
                    segment = route[i:i+k]
                    # Remove segment
                    # Prev node is route[i-1] if i>0 else 0
                    # Next node is route[i+k] if i+k<n else 0
                    prev_node = route[i-1] if i > 0 else 0
                    next_node = route[i+k] if i+k < n else 0
                    
                    old_cost = dist_matrix[prev_node, segment[0]] + dist_matrix[segment[-1], next_node]
                    
                    # Try inserting segment at all positions in the remaining route
                    # Remaining route is route[:i] + route[i+k:]
                    # We insert between nodes.
                    # Positions are 0..len(remaining).
                    # If pos is 0, insert before route[0] (if i!=0) or at start.
                    # This is tricky with indices.
                    
                    # Simplified Or-Opt: just try moving to a few promising spots?
                    # Or just do full simulation for Or-Opt moves as they are less frequent?
                    # Given time constraints, let's stick to 2-opt as the primary driver for speed.
                    # Or-Opt is O(n^2 * n) which is O(n^3). 2-opt is O(n^2).
                    # With n~25, n^3 is 15625 ops. Not bad.
                    
                    pass # Skip complex Or-Opt incremental for now to ensure 2-opt runs fast.

    # Relocation (Move customer from one route to another)
    for _pass in range(5):
        if time.time() > deadline - 2:
            break
        pass_improved = False
        for r1 in range(n_routes):
            if time.time() > deadline - 2:
                break
            if not routes[r1]: continue
            for pos1 in range(len(routes[r1])):
                cust = routes[r1][pos1]
                if not (1 <= cust <= n_customers):
                    continue
                for r2 in range(n_routes):
                    if r1 == r2:
                        continue
                    if not routes[r2]: continue
                    if route_load(routes[r2]) + demands[cust] > capacity:
                        continue
                    for pos2 in range(len(routes[r2]) + 1):
                        new_r1 = routes[r1][:pos1] + routes[r1][pos1+1:]
                        new_r2 = routes[r2][:pos2] + [cust] + routes[r2][pos2:]

                        if is_feasible(new_r1) and is_feasible(new_r2):
                            old_dist_part = route_dists[r1] + route_dists[r2]
                            new_dist_part = _compute_total_distance([new_r1, new_r2], dist_matrix)
                            if new_dist_part < old_dist_part - 1e-9:
                                routes[r1] = new_r1
                                routes[r2] = new_r2
                                route_dists[r1] = _compute_total_distance([new_r1], dist_matrix)
                                route_dists[r2] = _compute_total_distance([new_r2], dist_matrix)
                                current_dist += (route_dists[r1] + route_dists[r2]) - old_dist_part
                                pass_improved = True
                                break
                    if pass_improved:
                        break
                if pass_improved:
                    break
            if pass_improved:
                break

        if not pass_improved:
            break

    routes = [r for r in routes if len(r) > 0]
    return routes


def _simulated_annealing(routes, dist_matrix, instance, n_customers, stations_set,
                         battery_max, speed, recharge_rate, horizon, capacity, deadline):
    demands = instance["demands"]

    def get_total_dist(rts):
        return _compute_total_distance(rts, dist_matrix)

    def is_feasible(route):
        return _route_feasible(route, dist_matrix, instance, n_customers, stations_set,
                                battery_max, speed, recharge_rate, horizon, capacity)

    def route_load(route):
        return sum(demands[node] for node in route if 1 <= node <= n_customers)

    current_routes = [r[:] for r in routes]
    current_dist = get_total_dist(current_routes)
    best_routes = [r[:] for r in current_routes]
    best_dist = current_dist

    start_temp = 5.0
    end_temp = 0.05
    max_iters = 2000
    cooling_rate = (end_temp / start_temp) ** (1.0 / max(1, max_iters))
    temp = start_temp

    start_time = time.time()
    for iteration in range(max_iters):
        if time.time() > deadline:
            break

        new_routes = [r[:] for r in current_routes]
        perturbation_type = random.random()

        if perturbation_type < 0.5:
            if len(new_routes) > 0:
                r1 = random.randint(0, len(new_routes) - 1)
                if len(new_routes[r1]) > 0:
                    pos1 = random.randint(0, len(new_routes[r1]) - 1)
                    cust = new_routes[r1][pos1]
                    if 1 <= cust <= n_customers:
                        new_routes[r1].pop(pos1)
                        valid_targets = [r for r in range(len(new_routes))
                                         if route_load(new_routes[r]) + demands[cust] <= capacity]
                        if valid_targets:
                            r2 = random.choice(valid_targets)
                            pos2 = random.randint(0, len(new_routes[r2]))
                            new_routes[r2].insert(pos2, cust)
        elif perturbation_type < 0.8:
            if len(new_routes) >= 2:
                r1 = random.randint(0, len(new_routes) - 1)
                r2 = random.randint(0, len(new_routes) - 1)
                if r1 != r2 and len(new_routes[r1]) > 0 and len(new_routes[r2]) > 0:
                    pos1 = random.randint(0, len(new_routes[r1]) - 1)
                    pos2 = random.randint(0, len(new_routes[r2]) - 1)
                    cust1 = new_routes[r1][pos1]
                    cust2 = new_routes[r2][pos2]
                    if 1 <= cust1 <= n_customers and 1 <= cust2 <= n_customers:
                        load1 = route_load(new_routes[r1])
                        load2 = route_load(new_routes[r2])
                        if load1 - demands[cust1] + demands[cust2] <= capacity and \
                           load2 - demands[cust2] + demands[cust1] <= capacity:
                            new_routes[r1][pos1] = cust2
                            new_routes[r2][pos2] = cust1
        else:
            if len(new_routes) > 0:
                r1 = random.randint(0, len(new_routes) - 1)
                if len(new_routes[r1]) >= 2:
                    i = random.randint(0, len(new_routes[r1]) - 2)
                    j = random.randint(i + 1, len(new_routes[r1]) - 1)
                    new_routes[r1][i:j+1] = new_routes[r1][i:j+1][::-1]

        new_routes = [r for r in new_routes if len(r) > 0]
        all_feasible = True
        served = set()
        for route in new_routes:
            if not is_feasible(route):
                all_feasible = False
                break
            for node in route:
                if 1 <= node <= n_customers:
                    if node in served:
                        all_feasible = False
                        break
                    served.add(node)
            if not all_feasible:
                break

        if all_feasible and served == set(range(1, n_customers + 1)):
            new_dist = get_total_dist(new_routes)
            delta = new_dist - current_dist
            if delta < 0 or random.random() < math.exp(-delta / max(temp, 1e-10)):
                current_routes = new_routes
                current_dist = new_dist
                if current_dist < best_dist:
                    best_routes = [r[:] for r in current_routes]
                    best_dist = current_dist

        temp *= cooling_rate

    return best_routes


def _cluster_and_solve(instance, dist_matrix, n_customers, stations_set,
                        battery_max, speed, recharge_rate, horizon, capacity,
                        n_clusters=5, seed=0):
    random.seed(seed)
    coords = np.array(instance["coords"], dtype=np.float64)
    demands = instance["demands"]
    due = instance["due"]

    customer_coords = coords[1:n_customers+1]
    n_cust = n_customers

    centers = []
    remaining = list(range(n_cust))
    first = random.choice(remaining)
    centers.append(customer_coords[first].copy())
    remaining.remove(first)

    while len(centers) < n_clusters and remaining:
        max_dist = -1
        max_idx = 0
        for idx, cust_idx in enumerate(remaining):
            min_d = min(np.linalg.norm(customer_coords[cust_idx] - c) for c in centers)
            if min_d > max_dist:
                max_dist = min_d
                max_idx = idx
        centers.append(customer_coords[remaining[max_idx]].copy())
        remaining.pop(max_idx)

    clusters = [[] for _ in range(n_clusters)]
    for cust_idx in range(n_cust):
        min_d = float('inf')
        best_c = 0
        for c_idx, center in enumerate(centers):
            d = np.linalg.norm(customer_coords[cust_idx] - center)
            if d < min_d:
                min_d = d
                best_c = c_idx
        clusters[best_c].append(cust_idx + 1)

    routes = []
    for cluster in clusters:
        if not cluster:
            continue
        cluster_sorted = sorted(cluster, key=lambda v: due[v])
        sub_routes = []
        current_sub = []
        current_load = 0
        for cust in cluster_sorted:
            if current_load + demands[cust] > capacity:
                if current_sub:
                    sub_routes.append(current_sub)
                current_sub = [cust]
                current_load = demands[cust]
            else:
                current_sub.append(cust)
                current_load += demands[cust]
        if current_sub:
            sub_routes.append(current_sub)

        for sub in sub_routes:
            if len(sub) <= 12:
                tour = _solve_tsp_dp(sub, dist_matrix, max_points=12)
                route = [sub[i] for i in tour]
            else:
                route = sub

            optimized, feasible = _insert_station_optimal(
                route, dist_matrix, instance, n_customers, stations_set,
                battery_max, speed, recharge_rate, horizon, capacity)
            if feasible:
                routes.append(optimized)
            else:
                for cust in sub:
                    test = route + [cust]
                    opt, feas = _insert_station_optimal(
                        test, dist_matrix, instance, n_customers, stations_set,
                        battery_max, speed, recharge_rate, horizon, capacity)
                    if feas:
                        route = opt
                    else:
                        if route:
                            routes.append(route)
                        route = [cust]
                if route:
                    routes.append(route)

    return routes


def _construct_nearest_neighbor(instance, dist_matrix, n_customers, stations_set,
                                  battery_max, speed, recharge_rate, horizon, capacity,
                                  seed=0, strategy="due"):
    random.seed(seed)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]

    customers = list(range(1, n_customers + 1))
    if strategy == "due":
        customers.sort(key=lambda c: due[c])
    else:
        random.shuffle(customers)

    routes = []
    unserved = set(customers)
    route_loads = []

    while unserved:
        route = []
        load = 0
        current = 0
        t = 0.0
        battery = battery_max

        while unserved:
            best_cust = None
            best_dist = float('inf')
            for cust in unserved:
                if load + demands[cust] > capacity:
                    continue
                d = dist_matrix[current, cust]
                new_t = t + d / speed
                new_battery = battery - d
                if new_battery < -1e-9:
                    continue
                if new_t < ready[cust]:
                    new_t = ready[cust]
                if new_t > due[cust] + 1e-9:
                    continue
                if d < best_dist:
                    best_dist = d
                    best_cust = cust

            if best_cust is None:
                break

            route.append(best_cust)
            unserved.remove(best_cust)
            load += demands[best_cust]

            d = dist_matrix[current, best_cust]
            t += d / speed
            battery -= d

            if t < ready[best_cust]:
                t = ready[best_cust]
            t += service[best_cust]
            current = best_cust

        if route:
            routes.append(route)
            route_loads.append(load)

    return routes


def plan(instance):
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    coords = np.array(instance["coords"], dtype=np.float64)
    dist_matrix = _build_dist_matrix(coords)

    n_customers = instance["n_customers"]
    stations_set = set(instance["stations"])
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    capacity = instance["capacity"]

    best_routes = None
    best_dist = float('inf')

    # Phase 1: Multiple construction heuristics
    # 1a: Savings-based construction with different seeds
    for i in range(10):
        if time.time() > deadline - 60:
            break
        routes = _savings_construct(instance, dist_matrix, n_customers, stations_set,
                                    battery_max, speed, recharge_rate, horizon, capacity,
                                    seed=i)
        valid, dist = _validate_solution(routes, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = [r[:] for r in routes]

    # 1b: Nearest neighbor with due-date priority
    for i in range(3):
        if time.time() > deadline - 55:
            break
        routes = _construct_nearest_neighbor(instance, dist_matrix, n_customers, stations_set,
                                              battery_max, speed, recharge_rate, horizon,
                                              capacity, seed=i, strategy="due")
        valid, dist = _validate_solution(routes, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = [r[:] for r in routes]

    # 1c: Cluster and solve
    for i in range(3):
        if time.time() > deadline - 50:
            break
        for n_clusters in [4, 5, 6]:
            routes = _cluster_and_solve(instance, dist_matrix, n_customers, stations_set,
                                         battery_max, speed, recharge_rate, horizon, capacity,
                                         n_clusters=n_clusters, seed=i)
            valid, dist = _validate_solution(routes, instance)
            if valid and dist < best_dist:
                best_dist = dist
                best_routes = [r[:] for r in routes]

    # Fallback if all constructions failed
    if best_routes is None:
        best_routes = [[v] for v in range(1, n_customers + 1)]
        best_dist = _compute_total_distance(best_routes, dist_matrix)

    # Phase 2: Advanced Local Search (Optimized)
    # Allocate significant time here as it is the main improvement driver
    ls_deadline = deadline - 20
    if time.time() < ls_deadline:
        improved = _advanced_local_search(best_routes, dist_matrix, instance, n_customers,
                                          stations_set, battery_max, speed, recharge_rate,
                                          horizon, capacity, ls_deadline)
        valid, dist = _validate_solution(improved, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = improved

    # Phase 3: Simulated Annealing
    # Use remaining time, but ensure at least 5s buffer
    sa_deadline = deadline - 5
    if time.time() < sa_deadline:
        sa_routes = _simulated_annealing(best_routes, dist_matrix, instance, n_customers,
                                          stations_set, battery_max, speed, recharge_rate,
                                          horizon, capacity, sa_deadline)
        valid, dist = _validate_solution(sa_routes, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = sa_routes

    # Phase 4: Final Polish
    # Quick final local search
    final_deadline = deadline - 1
    if time.time() < final_deadline:
        improved = _advanced_local_search(best_routes, dist_matrix, instance, n_customers,
                                          stations_set, battery_max, speed, recharge_rate,
                                          horizon, capacity, final_deadline)
        valid, dist = _validate_solution(improved, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = improved

    # Final validation
    valid, dist = _validate_solution(best_routes, instance)
    if not valid:
        best_routes = [[v] for v in range(1, n_customers + 1)]

    return best_routes
