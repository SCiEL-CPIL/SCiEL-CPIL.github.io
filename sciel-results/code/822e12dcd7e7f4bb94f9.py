"""EV routing with time windows and battery constraints - model-and-optimize approach."""
import numpy as np
import time
import random
import math
from scipy.optimize import linear_sum_assignment

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


def _route_distance(route, dist_matrix):
    """Compute total distance of a route including depot legs."""
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
    """Fast feasibility check with early termination."""
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
        # Fallback: greedy
        return list(range(n))

    # Full bitmask DP
    full_mask = (1 << n) - 1
    INF = float('inf')
    # dp[mask][i] = min cost to start at 0, visit exactly set mask, end at i
    dp = [[INF] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]
    dp[1][0] = 0  # start at 0 (first point)

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

    # Find best ending point, then add return to 0
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

    # Reconstruct
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
    """
    Construct routes using Clarke-Wright Savings heuristic with time-window awareness.
    """
    random.seed(seed)
    demands = instance["demands"]
    due = instance["due"]
    ready = instance["ready"]
    
    # Initialize with each customer as a separate route
    routes = [[i] for i in range(1, n_customers + 1)]
    route_loads = [demands[i] for i in range(1, n_customers + 1)]
    
    # Calculate savings for all pairs
    savings = []
    for i in range(1, n_customers + 1):
        for j in range(i + 1, n_customers + 1):
            s = dist_matrix[0, i] + dist_matrix[0, j] - dist_matrix[i, j]
            if s > 1e-6:
                # Add time-window penalty: prefer merging customers with compatible time windows
                tw_penalty = abs(ready[i] - ready[j]) * 0.01
                savings.append((s - tw_penalty, i, j))
    
    # Sort savings in descending order
    savings.sort(key=lambda x: x[0], reverse=True)
    
    # Map customer to route index
    cust_to_route = {i: i - 1 for i in range(1, n_customers + 1)}
    
    # Process savings
    for s, i, j in savings:
        ri = cust_to_route[i]
        rj = cust_to_route[j]
        
        if ri == rj:
            continue
            
        route_i = routes[ri]
        route_j = routes[rj]
        
        if not route_i or not route_j:
            continue
        
        # Try all valid merge orientations
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
        
        # Check capacity
        if route_loads[ri] + route_loads[rj] > capacity:
            continue
            
        # Check feasibility with fast check
        if _is_route_feasible_fast(merged, dist_matrix, instance, n_customers,
                                    stations_set, battery_max, speed, recharge_rate,
                                    horizon, capacity,
                                    route_load=route_loads[ri] + route_loads[rj]):
            # Merge
            routes[ri] = merged
            routes[rj] = []
            route_loads[ri] += route_loads[rj]
            
            for cust in merged:
                if 1 <= cust <= n_customers:
                    cust_to_route[cust] = ri

    valid_routes = [r for r in routes if r]
    return valid_routes


def _optimize_route_with_stations(route, dist_matrix, instance, n_customers,
                                   stations_set, battery_max, speed, recharge_rate,
                                   horizon, capacity):
    """Try to insert charging stations to make a route feasible."""
    if _is_route_feasible_fast(route, dist_matrix, instance, n_customers, stations_set,
                                battery_max, speed, recharge_rate, horizon, capacity):
        return route, True

    # Try inserting one or two stations at best positions
    best_new_route = None
    best_cost = float('inf')
    
    # Try single station insertion
    for station in stations_set:
        for pos in range(len(route) + 1):
            new_route = route[:pos] + [station] + route[pos:]
            if _is_route_feasible_fast(new_route, dist_matrix, instance, n_customers,
                                        stations_set, battery_max, speed, recharge_rate,
                                        horizon, capacity):
                cost = _route_distance(new_route, dist_matrix)
                if cost < best_cost:
                    best_cost = cost
                    best_new_route = new_route
    
    # Try two station insertion if single didn't work
    if best_new_route is None:
        for st1 in stations_set:
            for st2 in stations_set:
                if st1 == st2:
                    continue
                for pos1 in range(len(route) + 1):
                    for pos2 in range(pos1 + 1, len(route) + 2):
                        new_route = route[:pos1] + [st1] + route[pos1:pos2] + [st2] + route[pos2:]
                        if _is_route_feasible_fast(new_route, dist_matrix, instance, n_customers,
                                                    stations_set, battery_max, speed, recharge_rate,
                                                    horizon, capacity):
                            cost = _route_distance(new_route, dist_matrix)
                            if cost < best_cost:
                                best_cost = cost
                                best_new_route = new_route

    if best_new_route is not None:
        return best_new_route, True
    return route, False


def _two_opt(route, dist_matrix, instance, n_customers, stations_set,
             battery_max, speed, recharge_rate, horizon, capacity, max_iters=50):
    """2-opt improvement within a route."""
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
                # Only consider segments that don't split stations awkwardly
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
    """
    Advanced local search with 2-Opt, Or-Opt, and inter-route relocate/swap.
    Uses fast feasibility checks and incremental distance updates.
    """
    demands = instance["demands"]

    def get_total_dist(rts):
        return _compute_total_distance(rts, dist_matrix)

    def is_feasible(route):
        return _is_route_feasible_fast(route, dist_matrix, instance, n_customers, stations_set,
                                        battery_max, speed, recharge_rate, horizon, capacity)

    def route_load(route):
        return sum(demands[node] for node in route if 1 <= node <= n_customers)

    current_dist = get_total_dist(routes)
    
    # Phase 1: Intra-route 2-Opt and Or-Opt
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        for r_idx in range(len(routes)):
            if time.time() > deadline - 3:
                break
            route = routes[r_idx]
            if len(route) < 2:
                continue
            
            route_d = _route_distance(route, dist_matrix)
            
            # 2-Opt with incremental distance check
            for i in range(len(route) - 1):
                for j in range(i + 1, len(route)):
                    # Calculate distance change for 2-opt reversal
                    a = 0 if i == 0 else route[i-1]
                    b = route[i]
                    c = route[j]
                    d = 0 if j == len(route)-1 else route[j+1]
                    
                    old_dist = dist_matrix[a][b] + dist_matrix[c][d]
                    new_dist = dist_matrix[a][c] + dist_matrix[b][d]
                    
                    if new_dist < old_dist - 1e-9:
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        if is_feasible(new_route):
                            routes[r_idx] = new_route
                            current_dist += new_dist - old_dist
                            improved = True
                            break
                if improved:
                    break
            
            if improved:
                continue

            # Or-Opt: move segment of size 1, 2, 3
            for k in [1, 2, 3]:
                if len(route) <= k:
                    continue
                for i in range(len(route) - k + 1):
                    segment = route[i:i+k]
                    rest = route[:i] + route[i+k:]
                    for j in range(len(rest) + 1):
                        if j == i or (j > i and j - i == k):
                            continue  # Same position
                        new_route = rest[:j] + segment + rest[j:]
                        if new_route == route:
                            continue
                        if is_feasible(new_route):
                            new_total = get_total_dist([new_route] + routes[:r_idx] + routes[r_idx+1:])
                            if new_total < current_dist - 1e-9:
                                routes[r_idx] = new_route
                                current_dist = new_total
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break

    # Phase 2: Inter-route relocate
    for _pass in range(20):
        if time.time() > deadline - 1:
            break
        pass_improved = False
        for r1 in range(len(routes)):
            if time.time() > deadline - 1:
                break
            if len(routes[r1]) == 0:
                continue
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
                            old_part = _route_distance(routes[r1], dist_matrix) + _route_distance(routes[r2], dist_matrix)
                            new_part = _route_distance(new_r1, dist_matrix) + _route_distance(new_r2, dist_matrix)
                            if new_part < old_part - 1e-9:
                                routes[r1] = new_r1
                                routes[r2] = new_r2
                                current_dist += new_part - old_part
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

    # Phase 3: Inter-route swap
    for _pass in range(10):
        if time.time() > deadline - 1:
            break
        pass_improved = False
        for r1 in range(len(routes)):
            if time.time() > deadline - 1:
                break
            if len(routes[r1]) == 0:
                continue
            for pos1 in range(len(routes[r1])):
                cust1 = routes[r1][pos1]
                if not (1 <= cust1 <= n_customers):
                    continue
                for r2 in range(len(routes)):
                    if r1 == r2 or len(routes[r2]) == 0:
                        continue
                    for pos2 in range(len(routes[r2])):
                        cust2 = routes[r2][pos2]
                        if not (1 <= cust2 <= n_customers):
                            continue
                        if cust1 == cust2:
                            continue
                        load1 = route_load(routes[r1]) - demands[cust1] + demands[cust2]
                        load2 = route_load(routes[r2]) - demands[cust2] + demands[cust1]
                        if load1 > capacity or load2 > capacity:
                            continue
                        new_r1 = routes[r1][:]
                        new_r2 = routes[r2][:]
                        new_r1[pos1] = cust2
                        new_r2[pos2] = cust1
                        if is_feasible(new_r1) and is_feasible(new_r2):
                            old_part = _route_distance(routes[r1], dist_matrix) + _route_distance(routes[r2], dist_matrix)
                            new_part = _route_distance(new_r1, dist_matrix) + _route_distance(new_r2, dist_matrix)
                            if new_part < old_part - 1e-9:
                                routes[r1] = new_r1
                                routes[r2] = new_r2
                                current_dist += new_part - old_part
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
    """Simulated annealing to escape local optima."""
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

    start_temp = 8.0
    end_temp = 0.1
    max_iters = 3000
    cooling_rate = (end_temp / start_temp) ** (1.0 / max(1, max_iters))
    temp = start_temp

    for iteration in range(max_iters):
        if time.time() > deadline:
            break

        new_routes = [r[:] for r in current_routes]
        perturbation_type = random.random()

        if perturbation_type < 0.4:
            # Relocate a customer
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
        elif perturbation_type < 0.7:
            # Swap two customers
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
            # Reverse segment (2-opt perturbation)
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
    """Construct via clustering then solve each cluster's TSP optimally."""
    random.seed(seed)
    coords = np.array(instance["coords"], dtype=np.float64)
    demands = instance["demands"]
    due = instance["due"]

    customer_coords = coords[1:n_customers+1]
    n_cust = n_customers

    # Farthest point sampling for centers
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

    # Assign
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
        # Split cluster into sub-routes if capacity exceeded
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
                # Solve TSP optimally
                tour = _solve_tsp_dp(sub, dist_matrix, max_points=12)
                route = [sub[i] for i in tour]
            else:
                route = sub

            # Try to make feasible with station insertion
            optimized, feasible = _optimize_route_with_stations(
                route, dist_matrix, instance, n_customers, stations_set,
                battery_max, speed, recharge_rate, horizon, capacity)
            if feasible:
                routes.append(optimized)
            else:
                # Fallback: greedy insertion
                for cust in sub:
                    test = route + [cust]
                    opt, feas = _optimize_route_with_stations(
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
    """Construct routes using nearest-neighbor heuristic with time-window awareness."""
    random.seed(seed)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    
    unserved = list(range(1, n_customers + 1))
    random.shuffle(unserved)
    
    routes = []
    while unserved:
        # Start a new route from the first unserved customer
        start = unserved.pop(0)
        route = [start]
        load = demands[start]
        current = start
        battery = battery_max
        time_at = dist_matrix[0, start] / speed
        battery -= dist_matrix[0, start]
        
        if 1 <= start <= n_customers:
            if time_at < ready[start]:
                time_at = ready[start]
            if time_at > due[start]:
                # Can't serve this customer, start fresh
                route = [start]
                unserved.insert(0, start)
                break
            time_at += instance["service"][start]
        
        while unserved:
            # Find best next customer
            best_cust = None
            best_score = float('inf')
            best_time = 0
            best_battery = 0
            
            for cust in unserved:
                if load + demands[cust] > capacity:
                    continue
                
                d = dist_matrix[current, cust]
                t = time_at + d / speed
                b = battery - d
                
                if b < -1e-9:
                    continue
                
                if t < ready[cust]:
                    t = ready[cust]
                if t > due[cust]:
                    continue
                
                t += instance["service"][cust]
                
                # Check if we can return to depot
                d_back = dist_matrix[cust, 0]
                t_back = t + d_back / speed
                b_back = b - d_back
                if b_back < -1e-9 or t_back > horizon:
                    continue
                
                score = d + dist_matrix[cust, 0] * 0.5  # Distance + return penalty
                if score < best_score:
                    best_score = score
                    best_cust = cust
                    best_time = t
                    best_battery = b
            
            if best_cust is None:
                break
            
            unserved.remove(best_cust)
            route.append(best_cust)
            load += demands[best_cust]
            current = best_cust
            time_at = best_time
            battery = best_battery
        
        # Check if route is feasible (with potential station insertion)
        optimized, feasible = _optimize_route_with_stations(
            route, dist_matrix, instance, n_customers, stations_set,
            battery_max, speed, recharge_rate, horizon, capacity)
        
        if feasible:
            routes.append(optimized)
        else:
            # Fallback: keep only the first customer
            routes.append([route[0]])
            for cust in route[1:]:
                unserved.append(cust)
    
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
    # Savings-based construction with different seeds
    for i in range(5):
        if time.time() > deadline - 80:
            break
        routes = _savings_construct(instance, dist_matrix, n_customers, stations_set,
                                    battery_max, speed, recharge_rate, horizon, capacity,
                                    seed=i)
        valid, dist = _validate_solution(routes, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = [r[:] for r in routes]

    # Nearest-neighbor construction with different strategies
    for i in range(3):
        if time.time() > deadline - 70:
            break
        routes = _construct_nearest_neighbor(instance, dist_matrix, n_customers, stations_set,
                                              battery_max, speed, recharge_rate, horizon,
                                              capacity, seed=i, strategy="due")
        valid, dist = _validate_solution(routes, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = [r[:] for r in routes]

    # Fallback
    if best_routes is None:
        best_routes = [[v] for v in range(1, n_customers + 1)]
        best_dist = _compute_total_distance(best_routes, dist_matrix)

    # Phase 2: Advanced Local Search
    if time.time() < deadline - 20:
        improved = _advanced_local_search(best_routes, dist_matrix, instance, n_customers,
                                          stations_set, battery_max, speed, recharge_rate,
                                          horizon, capacity, deadline - 10)
        valid, dist = _validate_solution(improved, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = improved

    # Phase 3: Simulated Annealing
    if time.time() < deadline - 25:
        sa_routes = _simulated_annealing(best_routes, dist_matrix, instance, n_customers,
                                          stations_set, battery_max, speed, recharge_rate,
                                          horizon, capacity, deadline - 10)
        valid, dist = _validate_solution(sa_routes, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = sa_routes

    # Phase 4: Final polish with local search
    if time.time() < deadline - 3:
        improved = _advanced_local_search(best_routes, dist_matrix, instance, n_customers,
                                          stations_set, battery_max, speed, recharge_rate,
                                          horizon, capacity, deadline - 1)
        valid, dist = _validate_solution(improved, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = improved

    # Final validation
    valid, dist = _validate_solution(best_routes, instance)
    if not valid:
        best_routes = [[v] for v in range(1, n_customers + 1)]

    return best_routes
