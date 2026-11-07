"""EV routing with time windows and charging - structural redesign."""
import numpy as np
import math
import time
import random
import heapq

def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

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

    # Precompute pairwise distances
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

    # Precompute which stations are feasible from each node with full battery
    # and time needed to go from node to station and back to depot
    station_set = set(stations)

    def simulate_route(route):
        """Simulate a route, return (total_distance, valid) or (inf, False)."""
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
        """
        Given a sequence of customer IDs, insert charging stops as needed.
        Returns (route_list, total_distance) or (None, inf) if infeasible.
        """
        if not cust_seq:
            return [], 0.0

        route = []
        t = 0.0
        bat = battery_cap
        load = 0
        prev = 0
        total_dist = 0.0

        for cust in cust_seq:
            # Check if we can go directly
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
                # Need to charge: find best station
                best_st = None
                best_cost = float('inf')
                best_t_cust = 0.0
                best_bat_after = 0.0
                best_total = 0.0

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
                    # Cost: extra distance + time penalty
                    extra_dist = d_to_st + d_st_cust - d_direct
                    t_after = t_cust + service[cust]
                    bat_after = bat_st - d_st_cust
                    load_new = load + demands[cust]
                    if load_new > capacity:
                        continue
                    # Prefer stations that minimize total distance
                    if extra_dist < best_cost - 1e-9:
                        best_cost = extra_dist
                        best_st = st
                        best_t_cust = t_after
                        best_bat_after = bat_after
                        best_total = extra_dist

                if best_st is None:
                    return None, float('inf')

                route.append(best_st)
                route.append(cust)
                t = best_t_cust
                bat = best_bat_after
                load += demands[cust]
                total_dist += best_total
                prev = cust

        # Return to depot
        d_ret = dist[prev][0]
        if bat >= d_ret - 1e-9:
            t_ret = t + d_ret / speed
            if t_ret > horizon + 1e-9:
                return None, float('inf')
            total_dist += d_ret
        else:
            # Need to charge before returning
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

    # Sort customers by due time for initial construction
    customers = list(range(1, n_customers + 1))
    customers.sort(key=lambda v: (due[v], ready[v]))

    # Clustering-based construction: divide customers into groups by proximity
    def cluster_customers(cust_list, n_clusters):
        """Simple k-means-like clustering on 2D coords."""
        if n_clusters >= len(cust_list):
            return [[c] for c in cust_list]

        # Initialize cluster centers using spread-out customers
        centers = []
        shuffled = cust_list[:]
        random.shuffle(shuffled)
        centers.append(shuffled[0])
        for c in shuffled[1:]:
            min_dist = min(math.sqrt((coords[c][0]-coords[cc][0])**2 + (coords[c][1]-coords[cc][1])**2) for cc in centers)
            if min_dist > 15.0:
                centers.append(c)
            if len(centers) >= n_clusters:
                break

        # Assign customers to nearest center
        assignments = [[] for _ in range(len(centers))]
        for c in cust_list:
            best_idx = 0
            best_d = float('inf')
            for idx, cc in enumerate(centers):
                d = math.sqrt((coords[c][0]-coords[cc][0])**2 + (coords[c][1]-coords[cc][1])**2)
                if d < best_d:
                    best_d = d
                    best_idx = idx
            assignments[best_idx].append(c)

        return [a for a in assignments if a]

    def construct_routes_from_clusters(clusters):
        """Build routes from customer clusters using greedy insertion."""
        routes = []
        for cluster in clusters:
            if not cluster:
                continue
            # Sort by angle from depot for initial ordering
            cluster_sorted = sorted(cluster, key=lambda c: math.atan2(coords[c][1]-coords[0][1], coords[c][0]-coords[0][0]))

            # Try different starting points and build route
            best_seq = None
            best_dist = float('inf')

            for start_idx in range(min(3, len(cluster_sorted))):
                seq = [cluster_sorted[start_idx]]
                remaining = [c for i, c in enumerate(cluster_sorted) if i != start_idx]
                current_load = demands[cluster_sorted[start_idx]]

                while remaining:
                    best_next = None
                    best_metric = float('inf')
                    last = seq[-1]

                    for c in remaining:
                        if current_load + demands[c] > capacity:
                            continue
                        # Quick feasibility check
                        d = dist[last][c]
                        if d > battery_cap:
                            continue
                        new_seq = seq + [c]
                        r, d_total = build_route_with_charging(new_seq)
                        if r is not None:
                            # Also check return
                            r_full, d_full = build_route_with_charging(new_seq)
                            if r_full is not None and d_full < best_metric:
                                best_metric = d_full
                                best_next = c

                    if best_next is None:
                        # Try with charging
                        for c in remaining:
                            if current_load + demands[c] > capacity:
                                continue
                            new_seq = seq + [c]
                            r, d_total = build_route_with_charging(new_seq)
                            if r is not None:
                                if d_total < best_metric:
                                    best_metric = d_total
                                    best_next = c

                    if best_next is None:
                        break
                    seq.append(best_next)
                    remaining.remove(best_next)
                    current_load += demands[best_next]

                r, d_total = build_route_with_charging(seq)
                if r is not None and d_total < best_dist:
                    best_dist = d_total
                    best_seq = seq

            if best_seq is not None:
                r, _ = build_route_with_charging(best_seq)
                if r is not None:
                    routes.append(r)
                else:
                    # Fallback: individual routes
                    for c in best_seq:
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

    def total_distance(routes):
        return sum(simulate_route(r)[0] for r in routes)

    # Try multiple cluster counts and pick the best
    best_routes = None
    best_total = float('inf')

    n_cust = n_customers
    cluster_counts = [3, 4, 5, 6, 7, 8]
    if n_cust <= 15:
        cluster_counts = [2, 3, 4, 5]

    for nc in cluster_counts:
        if time.time() > deadline - 20.0:
            break
        clusters = cluster_customers(customers, nc)
        routes = construct_routes_from_clusters(clusters)

        # Verify all customers covered
        covered = set()
        for r in routes:
            for node in r:
                if 1 <= node <= n_customers:
                    covered.add(node)

        if covered != set(range(1, n_customers + 1)):
            # Add missing
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
        # Fallback: each customer gets its own route
        best_routes = []
        for c in range(1, n_customers + 1):
            sr, _ = build_route_with_charging([c])
            if sr is not None:
                best_routes.append(sr)
            else:
                best_routes.append([c])
        best_total = total_distance(best_routes)

    routes = best_routes

    # Local search: relocate and swap
    def try_relocate(routes, i, j, k, pos):
        """Try to move customer at position j in route i to position pos in route k."""
        cust = routes[i][j]
        if not (1 <= cust <= n_customers):
            return False

        # Remove from route i
        new_i_seq = [c for idx, c in enumerate(routes[i]) if idx != j]
        # Insert into route k
        new_k_seq = routes[k][:pos] + [cust] + routes[k][pos:]

        r_i, d_i = build_route_with_charging(new_i_seq)
        r_k, d_k = build_route_with_charging(new_k_seq)

        if r_i is None or r_k is None:
            return False

        # Check no duplicate customers
        set_i = set(c for c in r_i if 1 <= c <= n_customers)
        set_k = set(c for c in r_k if 1 <= c <= n_customers)
        if set_i & set_k:
            return False

        old_dist = simulate_route(routes[i])[0] + simulate_route(routes[k])[0]
        new_dist = d_i + d_k

        if new_dist < old_dist - 0.1:
            routes[i] = r_i if r_i else []
            routes[k] = r_k
            # Remove empty routes
            if not routes[i]:
                del routes[i]
                if k > i:
                    k -= 1
            return True
        return False

    def try_swap(routes, i, j, k, pos):
        """Try to swap customer at position j in route i with customer at position pos in route k."""
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

    # 2-opt within each route
    def two_opt_route(routes, i):
        cust_seq = [c for c in routes[i] if 1 <= c <= n_customers]
        if len(cust_seq) < 2:
            return False

        best_seq = cust_seq[:]
        best_route, best_d = build_route_with_charging(cust_seq)
        if best_route is None:
            return False

        improved = False
        for _ in range(5):  # limited iterations
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

    # Run local search
    if time.time() < deadline - 15.0:
        improved = True
        max_iter = 50
        it = 0
        while improved and it < max_iter and time.time() < deadline - 10.0:
            improved = False
            it += 1

            # 2-opt within routes
            for i in range(len(routes)):
                if time.time() > deadline - 10.0:
                    break
                if two_opt_route(routes, i):
                    improved = True

            # Relocate
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

            # Swap
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

    # Final validation and cleanup
    valid_routes = []
    covered = set()
    for r in routes:
        d, valid = simulate_route(r)
        if valid:
            valid_routes.append(r)
            for node in r:
                if 1 <= node <= n_customers:
                    covered.add(node)

    # Add any missing customers
    for cust in range(1, n_customers + 1):
        if cust not in covered:
            sr, _ = build_route_with_charging([cust])
            if sr is not None:
                valid_routes.append(sr)
            else:
                valid_routes.append([cust])
            covered.add(cust)

    return valid_routes
