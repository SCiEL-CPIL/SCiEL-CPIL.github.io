"""EV routing with time windows and battery constraints - LNS with DP re-insertion."""
import numpy as np
import time
import random
import math

DEADLINE_SECONDS = 142.0


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

    for node in route:
        d = dist_matrix[current_node, node]
        time_at += d / speed
        battery -= d
        distance += d

        if battery < -1e-9:
            return False, 0.0, 0.0, 0.0

        if 1 <= node <= n_customers:
            if time_at < ready[node]:
                time_at = ready[node]
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
    time_at += d / speed
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


def _route_distance(route, dist_matrix):
    total = 0.0
    prev = 0
    for node in route:
        total += dist_matrix[prev, node]
        prev = node
    total += dist_matrix[prev, 0]
    return total


def _is_route_feasible_fast(route, dist_matrix, instance, n_customers, stations_set,
                             battery_max, speed, recharge_rate, horizon, capacity,
                             route_load=None):
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]

    if route_load is None:
        total_demand = 0
        for node in route:
            if 1 <= node <= n_customers:
                total_demand += demands[node]
        if total_demand > capacity:
            return False
    elif route_load > capacity:
        return False

    battery = battery_max
    time_at = 0.0
    current_node = 0

    for node in route:
        d = dist_matrix[current_node, node]
        time_at += d / speed
        battery -= d

        if battery < -1e-9:
            return False

        if 1 <= node <= n_customers:
            if time_at < ready[node]:
                time_at = ready[node]
            if time_at > due[node] + 1e-9:
                return False
            time_at += service[node]
        elif node in stations_set:
            energy_needed = battery_max - battery
            if energy_needed > 0:
                time_at += energy_needed * recharge_rate
            battery = battery_max

        current_node = node

    d = dist_matrix[current_node, 0]
    time_at += d / speed
    battery -= d

    if battery < -1e-9:
        return False
    if time_at > horizon + 1e-9:
        return False

    return True


def _savings_construct(instance, dist_matrix, n_customers, stations_set,
                       battery_max, speed, recharge_rate, horizon, capacity,
                       seed=0):
    random.seed(seed)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]

    routes = [[i] for i in range(1, n_customers + 1)]
    route_loads = [demands[i] for i in range(1, n_customers + 1)]

    savings = []
    for i in range(1, n_customers + 1):
        for j in range(i + 1, n_customers + 1):
            s = dist_matrix[0, i] + dist_matrix[0, j] - dist_matrix[i, j]
            if s > 1e-6:
                tw_penalty = abs(ready[i] - ready[j]) * 0.01
                savings.append((s - tw_penalty, i, j))

    savings.sort(key=lambda x: x[0], reverse=True)

    cust_to_route = {i: i - 1 for i in range(1, n_customers + 1)}

    for s, i, j in savings:
        ri = cust_to_route[i]
        rj = cust_to_route[j]

        if ri == rj:
            continue

        route_i = routes[ri]
        route_j = routes[rj]

        if not route_i or not route_j:
            continue

        merged = None
        if route_i[-1] == i and route_j[0] == j:
            merged = route_i + route_j
        elif route_i[0] == i and route_j[-1] == j:
            merged = route_i + route_j[::-1]
        elif route_i[-1] == i and route_j[-1] == j:
            merged = route_i + route_j[::-1]
        elif route_i[0] == i and route_j[0] == j:
            merged = route_i[::-1] + route_j
        else:
            continue

        if merged is None:
            continue

        if route_loads[ri] + route_loads[rj] > capacity:
            continue

        if _is_route_feasible_fast(merged, dist_matrix, instance, n_customers,
                                    stations_set, battery_max, speed, recharge_rate,
                                    horizon, capacity,
                                    route_load=route_loads[ri] + route_loads[rj]):
            routes[ri] = merged
            routes[rj] = []
            route_loads[ri] += route_loads[rj]

            for cust in merged:
                if 1 <= cust <= n_customers:
                    cust_to_route[cust] = ri

    valid_routes = [r for r in routes if r]
    return valid_routes


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

    return routes


def _solve_tsp_dp(points, dist_matrix, max_points=12):
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
            routes.append(route)

    return routes


def _matching_based_construct(instance, dist_matrix, n_customers, stations_set,
                               battery_max, speed, recharge_rate, horizon, capacity,
                               seed=0):
    random.seed(seed)
    demands = instance["demands"]

    customers = list(range(1, n_customers + 1))
    n = len(customers)

    pair_costs = {}
    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = customers[i], customers[j]
            s = dist_matrix[0, ci] + dist_matrix[0, cj] - dist_matrix[ci, cj]
            pair_costs[(ci, cj)] = s

    pairs = sorted(pair_costs.keys(), key=lambda p: pair_costs[p], reverse=True)

    matched = set()
    selected_pairs = []
    for ci, cj in pairs:
        if ci in matched or cj in matched:
            continue
        route = [ci, cj]
        if _is_route_feasible_fast(route, dist_matrix, instance, n_customers,
                                   stations_set, battery_max, speed, recharge_rate,
                                   horizon, capacity):
            selected_pairs.append((ci, cj))
            matched.add(ci)
            matched.add(cj)

    routes = []
    for ci, cj in selected_pairs:
        routes.append([ci, cj])

    for cust in customers:
        if cust not in matched:
            routes.append([cust])

    return routes


def _two_opt(route, dist_matrix, instance, n_customers, stations_set,
             battery_max, speed, recharge_rate, horizon, capacity, max_iters=30):
    if len(route) < 2:
        return route

    def is_feasible(r):
        return _is_route_feasible_fast(r, dist_matrix, instance, n_customers, stations_set,
                                       battery_max, speed, recharge_rate, horizon, capacity)

    def route_dist(r):
        return _route_distance(r, dist_matrix)

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


def _local_search(routes, dist_matrix, instance, n_customers, stations_set,
                  battery_max, speed, recharge_rate, horizon, capacity, deadline):
    demands = instance["demands"]

    def get_total_dist(rts):
        return _compute_total_distance(rts, dist_matrix)

    def is_feasible(route):
        return _is_route_feasible_fast(route, dist_matrix, instance, n_customers, stations_set,
                                       battery_max, speed, recharge_rate, horizon, capacity)

    def route_load(route):
        return sum(demands[node] for node in route if 1 <= node <= n_customers)

    current_dist = get_total_dist(routes)

    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for r_idx in range(len(routes)):
            if time.time() > deadline - 5:
                break
            route = routes[r_idx]
            if len(route) < 2:
                continue

            # 2-opt
            for i in range(len(route) - 1):
                for j in range(i + 1, len(route)):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if is_feasible(new_route):
                        new_dist = get_total_dist([new_route] + routes[:r_idx] + routes[r_idx+1:])
                        if new_dist < current_dist - 1e-9:
                            routes[r_idx] = new_route
                            current_dist = new_dist
                            improved = True
                            break
                if improved:
                    break

            if improved:
                continue

            # Or-opt
            for k in [1, 2, 3]:
                if len(route) <= k:
                    continue
                for i in range(len(route) - k + 1):
                    segment = route[i:i+k]
                    rest = route[:i] + route[i+k:]
                    for j in range(len(rest) + 1):
                        new_route = rest[:j] + segment + rest[j:]
                        if new_route == route:
                            continue
                        if is_feasible(new_route):
                            new_dist = get_total_dist([new_route] + routes[:r_idx] + routes[r_idx+1:])
                            if new_dist < current_dist - 1e-9:
                                routes[r_idx] = new_route
                                current_dist = new_dist
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break

    # Inter-route moves
    for _pass in range(10):
        if time.time() > deadline - 2:
            break
        pass_improved = False
        for r1 in range(len(routes)):
            if time.time() > deadline - 2:
                break
            for pos1 in range(len(routes[r1])):
                cust = routes[r1][pos1]
                if not (1 <= cust <= n_customers):
                    continue
                for r2 in range(len(routes)):
                    if r1 == r2:
                        continue
                    if route_load(routes[r2]) + demands[cust] > capacity:
                        continue
                    for pos2 in range(len(routes[r2]) + 1):
                        new_r1 = routes[r1][:pos1] + routes[r1][pos1+1:]
                        new_r2 = routes[r2][:pos2] + [cust] + routes[r2][pos2:]

                        if is_feasible(new_r1) and is_feasible(new_r2):
                            old_dist_part = _compute_total_distance([routes[r1], routes[r2]], dist_matrix)
                            new_dist_part = _compute_total_distance([new_r1, new_r2], dist_matrix)
                            if new_dist_part < old_dist_part - 1e-9:
                                routes[r1] = new_r1
                                routes[r2] = new_r2
                                current_dist = get_total_dist(routes)
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


def _dp_insert_customers(removed_customers, remaining_route, dist_matrix, instance,
                          n_customers, stations_set, battery_max, speed, recharge_rate,
                          horizon, capacity):
    """
    Use DP to find optimal insertion of removed_customers into remaining_route.
    State: (subset of removed customers placed, position in expanded route)
    This is a simplified version - we try all permutations of removed customers
    and find the best feasible insertion.
    """
    if not removed_customers:
        return remaining_route, True

    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]

    # For small number of removed customers, try all orderings
    n_removed = len(removed_customers)
    if n_removed > 5:
        # Too many for exhaustive, use greedy
        return _greedy_insert(removed_customers, remaining_route, dist_matrix, instance,
                              n_customers, stations_set, battery_max, speed, recharge_rate,
                              horizon, capacity)

    # Try all permutations
    from itertools import permutations

    best_route = None
    best_dist = float('inf')

    for perm in permutations(removed_customers):
        # Try inserting each customer at each position
        current_route = remaining_route[:]
        feasible = True
        for cust in perm:
            # Find best insertion position
            best_pos = None
            best_pos_dist = float('inf')
            for pos in range(len(current_route) + 1):
                candidate = current_route[:pos] + [cust] + current_route[pos:]
                if _is_route_feasible_fast(candidate, dist_matrix, instance, n_customers,
                                           stations_set, battery_max, speed, recharge_rate,
                                           horizon, capacity):
                    d = _route_distance(candidate, dist_matrix)
                    if d < best_pos_dist:
                        best_pos_dist = d
                        best_pos = pos
            if best_pos is None:
                feasible = False
                break
            current_route = current_route[:best_pos] + [cust] + current_route[best_pos:]

        if feasible:
            d = _route_distance(current_route, dist_matrix)
            if d < best_dist:
                best_dist = d
                best_route = current_route[:]

    if best_route is not None:
        return best_route, True
    return remaining_route, False


def _greedy_insert(removed_customers, remaining_route, dist_matrix, instance,
                    n_customers, stations_set, battery_max, speed, recharge_rate,
                    horizon, capacity):
    """Greedy insertion of removed customers."""
    demands = instance["demands"]
    current_route = remaining_route[:]

    # Sort by due date
    removed_sorted = sorted(removed_customers, key=lambda c: instance["due"][c])

    for cust in removed_sorted:
        best_pos = None
        best_pos_dist = float('inf')
        for pos in range(len(current_route) + 1):
            candidate = current_route[:pos] + [cust] + current_route[pos:]
            if _is_route_feasible_fast(candidate, dist_matrix, instance, n_customers,
                                       stations_set, battery_max, speed, recharge_rate,
                                       horizon, capacity):
                d = _route_distance(candidate, dist_matrix)
                if d < best_pos_dist:
                    best_pos_dist = d
                    best_pos = pos
        if best_pos is not None:
            current_route = current_route[:best_pos] + [cust] + current_route[best_pos:]
        else:
            # Can't insert, return failure
            return None, False

    return current_route, True


def _lns_search(routes, dist_matrix, instance, n_customers, stations_set,
                battery_max, speed, recharge_rate, horizon, capacity, deadline):
    """Large Neighborhood Search: destroy and repair."""
    demands = instance["demands"]

    def get_total_dist(rts):
        return _compute_total_distance(rts, dist_matrix)

    def is_feasible(route):
        return _is_route_feasible_fast(route, dist_matrix, instance, n_customers, stations_set,
                                       battery_max, speed, recharge_rate, horizon, capacity)

    def route_load(route):
        return sum(demands[node] for node in route if 1 <= node <= n_customers)

    current_routes = [r[:] for r in routes]
    current_dist = get_total_dist(current_routes)
    best_routes = [r[:] for r in current_routes]
    best_dist = current_dist

    start_temp = 3.0
    end_temp = 0.1
    max_iters = 500
    cooling_rate = (end_temp / start_temp) ** (1.0 / max(1, max_iters))
    temp = start_temp

    start_time = time.time()
    for iteration in range(max_iters):
        if time.time() > deadline:
            break

        # Destroy: remove random subset of customers from a random route
        new_routes = [r[:] for r in current_routes]
        r_idx = random.randint(0, len(new_routes) - 1)
        route = new_routes[r_idx]

        if len(route) < 3:
            continue

        # Remove 2-4 customers
        n_remove = random.randint(2, min(4, len(route)))
        removed = random.sample([c for c in route if 1 <= c <= n_customers], n_remove)
        remaining = [c for c in route if c not in removed]

        # Repair: try to re-insert removed customers
        new_route, feasible = _dp_insert_customers(removed, remaining, dist_matrix, instance,
                                                    n_customers, stations_set, battery_max,
                                                    speed, recharge_rate, horizon, capacity)

        if not feasible or new_route is None:
            # Try inserting into other routes
            success = False
            for cust in removed:
                best_target = None
                best_target_dist = float('inf')
                for t_idx in range(len(new_routes)):
                    if t_idx == r_idx:
                        continue
                    if route_load(new_routes[t_idx]) + demands[cust] > capacity:
                        continue
                    for pos in range(len(new_routes[t_idx]) + 1):
                        candidate = new_routes[t_idx][:pos] + [cust] + new_routes[t_idx][pos:]
                        if is_feasible(candidate):
                            d = _route_distance(candidate, dist_matrix)
                            if d < best_target_dist:
                                best_target_dist = d
                                best_target = (t_idx, pos)
                if best_target is not None:
                    t_idx, pos = best_target
                    new_routes[t_idx] = new_routes[t_idx][:pos] + [cust] + new_routes[t_idx][pos:]
                    success = True
                else:
                    success = False
                    break

            if not success:
                temp *= cooling_rate
                continue

            # Update route
            new_routes[r_idx] = remaining
            new_routes = [r for r in new_routes if len(r) > 0]
        else:
            new_routes[r_idx] = new_route
            new_routes = [r for r in new_routes if len(r) > 0]

        # Validate
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


def _simulated_annealing(routes, dist_matrix, instance, n_customers, stations_set,
                         battery_max, speed, recharge_rate, horizon, capacity, deadline):
    demands = instance["demands"]

    def get_total_dist(rts):
        return _compute_total_distance(rts, dist_matrix)

    def is_feasible(route):
        return _is_route_feasible_fast(route, dist_matrix, instance, n_customers, stations_set,
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
    # 1a: Savings-based construction
    for i in range(8):
        if time.time() > deadline - 60:
            break
        routes = _savings_construct(instance, dist_matrix, n_customers, stations_set,
                                    battery_max, speed, recharge_rate, horizon, capacity,
                                    seed=i)
        valid, dist = _validate_solution(routes, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = [r[:] for r in routes]

    # 1b: Nearest neighbor
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

    # 1d: Matching-based construction
    for i in range(5):
        if time.time() > deadline - 45:
            break
        routes = _matching_based_construct(instance, dist_matrix, n_customers, stations_set,
                                            battery_max, speed, recharge_rate, horizon, capacity,
                                            seed=i)
        valid, dist = _validate_solution(routes, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = [r[:] for r in routes]

    # Fallback
    if best_routes is None:
        best_routes = [[v] for v in range(1, n_customers + 1)]
        best_dist = _compute_total_distance(best_routes, dist_matrix)

    # Phase 2: Local Search
    if time.time() < deadline - 20:
        improved = _local_search(best_routes, dist_matrix, instance, n_customers,
                                  stations_set, battery_max, speed, recharge_rate,
                                  horizon, capacity, deadline - 15)
        valid, dist = _validate_solution(improved, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = improved

    # Phase 3: LNS with DP re-insertion
    if time.time() < deadline - 25:
        lns_routes = _lns_search(best_routes, dist_matrix, instance, n_customers,
                                  stations_set, battery_max, speed, recharge_rate,
                                  horizon, capacity, deadline - 15)
        valid, dist = _validate_solution(lns_routes, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = lns_routes

    # Phase 4: Simulated Annealing
    if time.time() < deadline - 20:
        sa_routes = _simulated_annealing(best_routes, dist_matrix, instance, n_customers,
                                          stations_set, battery_max, speed, recharge_rate,
                                          horizon, capacity, deadline - 10)
        valid, dist = _validate_solution(sa_routes, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = sa_routes

    # Phase 5: Final polish
    if time.time() < deadline - 5:
        improved = _local_search(best_routes, dist_matrix, instance, n_customers,
                                  stations_set, battery_max, speed, recharge_rate,
                                  horizon, capacity, deadline - 2)
        valid, dist = _validate_solution(improved, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = improved

    # Final validation
    valid, dist = _validate_solution(best_routes, instance)
    if not valid:
        best_routes = [[v] for v in range(1, n_customers + 1)]

    return best_routes
