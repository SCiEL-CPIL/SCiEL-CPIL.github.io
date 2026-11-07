"""EV routing with time windows and charging - DP on clustered sub-problems."""
import numpy as np
import math
import time
import random

def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0

    n_customers = instance["n_customers"]
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    n_nodes = len(coords)
    coords_arr = np.array(coords, dtype=float)
    dist = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            dx = coords_arr[i][0] - coords_arr[j][0]
            dy = coords_arr[i][1] - coords_arr[j][1]
            d = math.sqrt(dx * dx + dy * dy)
            dist[i][j] = d
            dist[j][i] = d

    station_set = set(stations)

    def simulate_route(route):
        if not route:
            return 0.0, True
        t = 0.0
        bat = battery_cap
        load = 0
        prev = 0
        total_dist = 0.0
        for node in route:
            d = dist[prev][node]
            total_dist += d
            if bat < d - 1e-9:
                return float('inf'), False
            t += d / speed
            bat -= d
            load += demands[node]
            if load > capacity:
                return float('inf'), False
            if 1 <= node <= n_customers:
                if t < ready[node]:
                    t = ready[node]
                if t > due[node] + 1e-9:
                    return float('inf'), False
                t += service[node]
            elif node in station_set:
                deficit = battery_cap - bat
                if deficit > 0:
                    t += deficit * recharge_rate
                bat = battery_cap
            prev = node
        d_ret = dist[prev][0]
        total_dist += d_ret
        if bat < d_ret - 1e-9:
            return float('inf'), False
        t += d_ret / speed
        if t > horizon + 1e-9:
            return float('inf'), False
        return total_dist, True

    def build_route_with_charging(cust_seq):
        if not cust_seq:
            return [], 0.0
        route = []
        t = 0.0
        bat = battery_cap
        load = 0
        prev = 0
        total_dist = 0.0
        for cust in cust_seq:
            d_direct = dist[prev][cust]
            if bat >= d_direct - 1e-9:
                t_arr = t + d_direct / speed
                if t_arr < ready[cust]:
                    t_arr = ready[cust]
                if t_arr > due[cust] + 1e-9:
                    return None, float('inf')
                t = t_arr + service[cust]
                bat -= d_direct
                load += demands[cust]
                if load > capacity:
                    return None, float('inf')
                total_dist += d_direct
                route.append(cust)
                prev = cust
            else:
                best_st = None
                best_cost = float('inf')
                best_t_cust = 0.0
                best_bat_after = 0.0
                for st in stations:
                    d_to_st = dist[prev][st]
                    if bat < d_to_st - 1e-9:
                        continue
                    t_st = t + d_to_st / speed
                    bat_at_st = bat - d_to_st
                    rt = max(0.0, (battery_cap - bat_at_st) * recharge_rate)
                    t_st += rt
                    bat_st = battery_cap
                    d_st_cust = dist[st][cust]
                    if bat_st < d_st_cust - 1e-9:
                        continue
                    t_cust = t_st + d_st_cust / speed
                    if t_cust < ready[cust]:
                        t_cust = ready[cust]
                    if t_cust > due[cust] + 1e-9:
                        continue
                    extra_dist = d_to_st + d_st_cust - d_direct
                    t_after = t_cust + service[cust]
                    bat_after = bat_st - d_st_cust
                    load_new = load + demands[cust]
                    if load_new > capacity:
                        continue
                    if extra_dist < best_cost - 1e-9:
                        best_cost = extra_dist
                        best_st = st
                        best_t_cust = t_after
                        best_bat_after = bat_after
                if best_st is None:
                    return None, float('inf')
                route.append(best_st)
                route.append(cust)
                t = best_t_cust
                bat = best_bat_after
                load += demands[cust]
                total_dist += best_cost
                prev = cust
        d_ret = dist[prev][0]
        if bat >= d_ret - 1e-9:
            t_ret = t + d_ret / speed
            if t_ret > horizon + 1e-9:
                return None, float('inf')
            total_dist += d_ret
        else:
            found = False
            for st in stations:
                d_to_st = dist[prev][st]
                if bat < d_to_st - 1e-9:
                    continue
                t_st = t + d_to_st / speed
                bat_at_st = bat - d_to_st
                rt = max(0.0, (battery_cap - bat_at_st) * recharge_rate)
                t_st += rt
                bat_st = battery_cap
                d_st_dep = dist[st][0]
                if bat_st < d_st_dep - 1e-9:
                    continue
                t_dep = t_st + d_st_dep / speed
                if t_dep > horizon + 1e-9:
                    continue
                total_dist += d_to_st + d_st_dep
                route.append(st)
                found = True
                break
            if not found:
                return None, float('inf')
        return route, total_dist

    def route_distance(route):
        if not route:
            return 0.0
        total = 0.0
        prev = 0
        for node in route:
            total += dist[prev][node]
            prev = node
        total += dist[prev][0]
        return total

    # K-means clustering
    def kmeans_cluster(cust_list, n_clusters, max_iter=20):
        if n_clusters >= len(cust_list):
            return [[c] for c in cust_list]
        random.seed(42)
        shuffled = cust_list[:]
        random.shuffle(shuffled)
        centers = [coords[c] for c in shuffled[:n_clusters]]
        assignments = [list(range(n_clusters)) for _ in range(n_clusters)]
        for _ in range(max_iter):
            new_assignments = [[] for _ in range(n_clusters)]
            for c in cust_list:
                best_idx = 0
                best_d = float('inf')
                for idx in range(n_clusters):
                    d = math.sqrt((coords[c][0]-centers[idx][0])**2 + (coords[c][1]-centers[idx][1])**2)
                    if d < best_d:
                        best_d = d
                        best_idx = idx
                new_assignments[best_idx].append(c)
            # Update centers
            for idx in range(n_clusters):
                if new_assignments[idx]:
                    cx = sum(coords[c][0] for c in new_assignments[idx]) / len(new_assignments[idx])
                    cy = sum(coords[c][1] for c in new_assignments[idx]) / len(new_assignments[idx])
                    centers[idx] = (cx, cy)
            if new_assignments == assignments:
                break
            assignments = new_assignments
        return [a for a in assignments if a]

    # DP for optimal route within a cluster
    def dp_optimal_route(cluster):
        """
        Exact DP for optimal route within a cluster.
        State: (mask, last_customer) -> (min_time, min_distance, battery_at_end, path)
        We try all permutations via bitmask DP.
        """
        if not cluster:
            return [], 0.0
        n = len(cluster)
        if n > 12:  # Too large for bitmask DP
            return None, float('inf')
        
        # Map cluster indices to actual customer IDs
        # dp[mask][i] = (min_total_dist, min_time, min_battery_remaining, path_list)
        # We want to minimize distance, but time and battery are constraints
        
        # Actually, let's use a different approach:
        # dp[mask][i] = set of feasible (time, battery, distance) states
        # For efficiency, we'll keep only the best (min distance) for each (time, battery) tradeoff
        # But that's complex. Let's just enumerate all permutations for small n.
        
        if n <= 8:
            # Try all permutations
            from itertools import permutations
            best_dist = float('inf')
            best_seq = None
            for perm in permutations(cluster):
                r, d = build_route_with_charging(list(perm))
                if r is not None and d < best_dist:
                    best_dist = d
                    best_seq = list(perm)
            if best_seq is not None:
                r, _ = build_route_with_charging(best_seq)
                return r, best_dist
            return None, float('inf')
        else:
            # Use DP with bitmask
            # dp[mask][last] = (min_dist, min_time, battery, path)
            # We minimize distance, with time and battery as constraints
            # Actually, we need to track time and battery carefully.
            # Let's use: dp[mask][last] = list of (dist, time, battery) Pareto-optimal states
            
            INF = float('inf')
            dp = {}
            # Initialize: single customer
            for i, c in enumerate(cluster):
                mask = 1 << i
                r, d = build_route_with_charging([c])
                if r is not None:
                    # Simulate to get time and battery
                    t = 0.0
                    bat = battery_cap
                    prev = 0
                    for node in r:
                        dd = dist[prev][node]
                        t += dd / speed
                        bat -= dd
                        if 1 <= node <= n_customers:
                            if t < ready[node]:
                                t = ready[node]
                            t += service[node]
                        elif node in station_set:
                            deficit = battery_cap - bat
                            if deficit > 0:
                                t += deficit * recharge_rate
                            bat = battery_cap
                        prev = node
                    dp[(mask, i)] = [(d, t, bat, [c])]
            
            # DP transitions
            for mask in range(1, 1 << n):
                for last in range(n):
                    if (mask, last) not in dp:
                        continue
                    for nxt in range(n):
                        if mask & (1 << nxt):
                            continue
                        new_mask = mask | (1 << nxt)
                        for (d, t, bat, path) in dp[(mask, last)]:
                            # Try adding cluster[nxt] after cluster[last]
                            new_seq = path + [cluster[nxt]]
                            r, d_new = build_route_with_charging(new_seq)
                            if r is not None:
                                # Check if this state is Pareto-optimal
                                new_state = (d_new, t, bat, new_seq)
                                # For simplicity, just keep the best distance
                                if (new_mask, nxt) not in dp:
                                    dp[(new_mask, nxt)] = [new_state]
                                else:
                                    # Keep only states with better distance or better time/battery
                                    existing = dp[(new_mask, nxt)]
                                    dominated = False
                                    for (ed, et, eb, ep) in existing:
                                        if ed <= d_new and et <= t and eb >= bat:
                                            dominated = True
                                            break
                                    if not dominated:
                                        # Remove dominated states
                                        dp[(new_mask, nxt)] = [s for s in existing 
                                                               if not (s[0] <= d_new and s[1] <= t and s[2] >= bat)]
                                        dp[(new_mask, nxt)].append(new_state)
            
            # Find best final state
            full_mask = (1 << n) - 1
            best_dist = float('inf')
            best_path = None
            for last in range(n):
                if (full_mask, last) in dp:
                    for (d, t, bat, path) in dp[(full_mask, last)]:
                        if d < best_dist:
                            best_dist = d
                            best_path = path
            if best_path is not None:
                r, _ = build_route_with_charging(best_path)
                return r, best_dist
            return None, float('inf')

    # Construct routes using clustering + DP
    def construct_with_dp(clusters):
        routes = []
        for cluster in clusters:
            if not cluster:
                continue
            r, d = dp_optimal_route(cluster)
            if r is not None:
                routes.append(r)
            else:
                # Fallback: split into smaller groups
                if len(cluster) > 8:
                    mid = len(cluster) // 2
                    for sub in [cluster[:mid], cluster[mid:]]:
                        r2, d2 = dp_optimal_route(sub)
                        if r2 is not None:
                            routes.append(r2)
                        else:
                            for c in sub:
                                sr, _ = build_route_with_charging([c])
                                if sr is not None:
                                    routes.append(sr)
                                else:
                                    routes.append([c])
                else:
                    for c in cluster:
                        sr, _ = build_route_with_charging([c])
                        if sr is not None:
                            routes.append(sr)
                        else:
                            routes.append([c])
        return routes

    # Greedy construction
    def greedy_construct():
        routes = []
        remaining = set(range(1, n_customers + 1))
        while remaining and time.time() < deadline - 5.0:
            best_route = None
            best_dist = float('inf')
            best_used = None
            start_candidates = sorted(remaining, key=lambda v: due[v])
            for start_cust in start_candidates[:8]:
                route_seq = [start_cust]
                used = {start_cust}
                current_load = demands[start_cust]
                while time.time() < deadline - 5.0:
                    best_next = None
                    best_metric = float('inf')
                    for c in remaining - used:
                        if current_load + demands[c] > capacity:
                            continue
                        new_seq = route_seq + [c]
                        test_route, d_total = build_route_with_charging(new_seq)
                        if test_route is not None and d_total < best_metric:
                            best_metric = d_total
                            best_next = c
                    if best_next is None:
                        break
                    route_seq.append(best_next)
                    used.add(best_next)
                    current_load += demands[best_next]
                final_route, d_total = build_route_with_charging(route_seq)
                if final_route is not None:
                    d = route_distance(final_route)
                    if d < best_dist:
                        best_dist = d
                        best_route = final_route
                        best_used = set(c for c in route_seq if 1 <= c <= n_customers)
            if best_route is None:
                for c in remaining:
                    sr, _ = build_route_with_charging([c])
                    if sr is not None:
                        routes.append(sr)
                        remaining.discard(c)
                        break
                else:
                    break
            else:
                routes.append(best_route)
                remaining -= best_used
        for c in remaining:
            sr, _ = build_route_with_charging([c])
            if sr is not None:
                routes.append(sr)
            else:
                routes.append([c])
        return routes

    def total_distance(routes):
        return sum(simulate_route(r)[0] for r in routes)

    best_routes = None
    best_total = float('inf')

    customers = list(range(1, n_customers + 1))

    # Strategy 1: K-means + DP
    for nc in [3, 4, 5, 6, 7]:
        if time.time() > deadline - 30.0:
            break
        clusters = kmeans_cluster(customers, nc)
        routes = construct_with_dp(clusters)
        covered = set()
        for r in routes:
            for node in r:
                if 1 <= node <= n_customers:
                    covered.add(node)
        if covered != set(range(1, n_customers + 1)):
            for c in set(range(1, n_customers + 1)) - covered:
                sr, _ = build_route_with_charging([c])
                if sr is not None:
                    routes.append(sr)
                else:
                    routes.append([c])
        td = total_distance(routes)
        if td < best_total:
            best_total = td
            best_routes = routes[:]

    # Strategy 2: Greedy
    if time.time() < deadline - 20.0:
        routes = greedy_construct()
        covered = set()
        for r in routes:
            for node in r:
                if 1 <= node <= n_customers:
                    covered.add(node)
        if covered != set(range(1, n_customers + 1)):
            for c in set(range(1, n_customers + 1)) - covered:
                sr, _ = build_route_with_charging([c])
                if sr is not None:
                    routes.append(sr)
                else:
                    routes.append([c])
        td = total_distance(routes)
        if td < best_total:
            best_total = td
            best_routes = routes[:]

    if best_routes is None:
        best_routes = []
        for c in range(1, n_customers + 1):
            sr, _ = build_route_with_charging([c])
            if sr is not None:
                best_routes.append(sr)
            else:
                best_routes.append([c])

    routes = best_routes

    # Local search
    def try_relocate(routes, i, j, k, pos):
        cust = routes[i][j]
        if not (1 <= cust <= n_customers):
            return False
        new_i_seq = [c for idx, c in enumerate(routes[i]) if idx != j]
        new_k_seq = routes[k][:pos] + [cust] + routes[k][pos:]
        r_i, d_i = build_route_with_charging(new_i_seq)
        r_k, d_k = build_route_with_charging(new_k_seq)
        if r_i is None or r_k is None:
            return False
        set_i = set(c for c in r_i if 1 <= c <= n_customers)
        set_k = set(c for c in r_k if 1 <= c <= n_customers)
        if set_i & set_k:
            return False
        old_dist = simulate_route(routes[i])[0] + simulate_route(routes[k])[0]
        new_dist = d_i + d_k
        if new_dist < old_dist - 0.1:
            routes[i] = r_i if r_i else []
            routes[k] = r_k
            if not routes[i]:
                del routes[i]
                if k > i:
                    k -= 1
            return True
        return False

    def try_swap(routes, i, j, k, pos):
        cust_i = routes[i][j]
        cust_k = routes[k][pos]
        if not (1 <= cust_i <= n_customers) or not (1 <= cust_k <= n_customers):
            return False
        new_i_seq = routes[i][:]
        new_k_seq = routes[k][:]
        new_i_seq[j] = cust_k
        new_k_seq[pos] = cust_i
        r_i, d_i = build_route_with_charging(new_i_seq)
        r_k, d_k = build_route_with_charging(new_k_seq)
        if r_i is None or r_k is None:
            return False
        set_i = set(c for c in r_i if 1 <= c <= n_customers)
        set_k = set(c for c in r_k if 1 <= c <= n_customers)
        if set_i & set_k:
            return False
        old_dist = simulate_route(routes[i])[0] + simulate_route(routes[k])[0]
        new_dist = d_i + d_k
        if new_dist < old_dist - 0.1:
            routes[i] = r_i
            routes[k] = r_k
            return True
        return False

    def two_opt_route(routes, i):
        cust_seq = [c for c in routes[i] if 1 <= c <= n_customers]
        if len(cust_seq) < 2:
            return False
        best_seq = cust_seq[:]
        best_route, best_d = build_route_with_charging(cust_seq)
        if best_route is None:
            return False
        improved = False
        for _ in range(5):
            changed = False
            for a in range(len(best_seq)):
                for b in range(a + 1, len(best_seq)):
                    new_seq = best_seq[:a] + best_seq[a:b+1][::-1] + best_seq[b+1:]
                    r, d = build_route_with_charging(new_seq)
                    if r is not None and d < best_d - 0.1:
                        best_d = d
                        best_seq = new_seq[:]
                        best_route = r
                        changed = True
                        improved = True
            if not changed:
                break
        if improved:
            routes[i] = best_route
        return improved

    if time.time() < deadline - 15.0:
        improved = True
        max_iter = 50
        it = 0
        while improved and it < max_iter and time.time() < deadline - 10.0:
            improved = False
            it += 1
            for i in range(len(routes)):
                if time.time() > deadline - 10.0:
                    break
                if two_opt_route(routes, i):
                    improved = True
            for i in range(len(routes)):
                if time.time() > deadline - 10.0:
                    break
                for j in range(len(routes[i])):
                    cust = routes[i][j]
                    if not (1 <= cust <= n_customers):
                        continue
                    for k in range(len(routes)):
                        if k == i or time.time() > deadline - 10.0:
                            continue
                        for pos in range(len(routes[k]) + 1):
                            if try_relocate(routes, i, j, k, pos):
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved and time.time() < deadline - 10.0:
                for i in range(len(routes)):
                    if time.time() > deadline - 10.0:
                        break
                    for j in range(len(routes[i])):
                        cust_i = routes[i][j]
                        if not (1 <= cust_i <= n_customers):
                            continue
                        for k in range(len(routes)):
                            if k == i or time.time() > deadline - 10.0:
                                continue
                            for pos in range(len(routes[k])):
                                if try_swap(routes, i, j, k, pos):
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break

    # Final validation
    valid_routes = []
    covered = set()
    for r in routes:
        d, valid = simulate_route(r)
        if valid:
            valid_routes.append(r)
            for node in r:
                if 1 <= node <= n_customers:
                    covered.add(node)
    for cust in range(1, n_customers + 1):
        if cust not in covered:
            sr, _ = build_route_with_charging([cust])
            if sr is not None:
                valid_routes.append(sr)
            else:
                valid_routes.append([cust])
            covered.add(cust)

    return valid_routes
