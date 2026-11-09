"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    coords = np.array(instance["coords"], dtype=np.float64)
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    # Precompute distance matrix with NumPy
    n_nodes = len(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    customers = list(range(1, n_customers + 1))
    station_set = set(stations)

    def fast_check_route(route):
        """Fast feasibility check with early termination. Returns (feasible, total_distance)."""
        if not route:
            return True, 0.0

        total_demand = sum(demands[v] for v in route)
        if total_demand > capacity:
            return False, 0.0

        t = 0.0
        bat = battery
        prev = 0
        total_dist = 0.0

        for v in route:
            d = dist[prev, v]
            bat -= d
            if bat < -1e-9:
                return False, 0.0
            t += d / speed
            total_dist += d

            if t < ready[v]:
                t = ready[v]
            if t > due[v] + 1e-9:
                return False, 0.0
            t += service[v]
            prev = v

        d = dist[prev, 0]
        bat -= d
        if bat < -1e-9:
            return False, 0.0
        t += d / speed
        total_dist += d

        if t > horizon + 1e-9:
            return False, 0.0

        return True, total_dist

    def total_route_distance(routes):
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += dist[0, route[0]]
            for i in range(len(route) - 1):
                total += dist[route[i], route[i + 1]]
            total += dist[route[-1], 0]
        return total

    def validate_routes(routes):
        """Check if routes are valid: all customers served exactly once, all feasible."""
        all_customers = []
        for route in routes:
            feasible, _ = fast_check_route(route)
            if not feasible:
                return False
            all_customers.extend(route)
        if len(all_customers) != len(set(all_customers)):
            return False
        if set(all_customers) != set(customers):
            return False
        return True

    # ---- Initial solution: one customer per route (always feasible) ----
    best_routes = [[v] for v in customers]
    best_distance = sum(dist[0, v] * 2 for v in customers)

    # ---- K-means spatial clustering + nearest-neighbor construction ----
    def kmeans_construct(k, seed=0):
        """Cluster customers spatially (K-means), then build each cluster's route
        by nearest-neighbour insertion with feasibility checks."""
        rng = random.Random(seed)
        if k > n_customers:
            k = n_customers

        # K-means on customer coordinates
        cust_coords = coords[1:n_customers + 1]
        # init: pick k distinct points
        indices = list(range(n_customers))
        rng.shuffle(indices)
        centers = cust_coords[indices[:k]].copy()

        for _ in range(15):
            # assign
            d_to_centers = np.linalg.norm(cust_coords[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
            labels = np.argmin(d_to_centers, axis=1)
            new_centers = np.zeros_like(centers)
            for c in range(k):
                mask = labels == c
                if mask.sum() > 0:
                    new_centers[c] = cust_coords[mask].mean(axis=0)
                else:
                    new_centers[c] = centers[c]
            if np.allclose(new_centers, centers):
                centers = new_centers
                break
            centers = new_centers

        # build each cluster as a route via nearest neighbour
        clusters = [list(np.where(labels == c)[0]) for c in range(k)]
        # convert to customer node ids
        cluster_ids = [[i + 1 for i in cl] for cl in clusters]

        routes = []
        for cl in cluster_ids:
            if not cl:
                continue
            remaining = list(cl)
            route = []
            prev = 0
            t = 0.0
            bat = battery
            load = 0
            while remaining:
                best_v = None
                best_score = float('inf')
                best_idx = -1
                for idx, v in enumerate(remaining):
                    d = dist[prev, v]
                    bat_after = bat - d
                    t_after = t + d / speed
                    if bat_after < -1e-9:
                        continue
                    if load + demands[v] > capacity:
                        continue
                    if t_after < ready[v]:
                        t_after = ready[v]
                    if t_after > due[v] + 1e-9:
                        continue
                    slack = due[v] - t_after
                    score = d - 0.02 * slack
                    if score < best_score:
                        best_score = score
                        best_v = v
                        best_idx = idx
                if best_v is None:
                    # can't place any more in this route; start a new route for the rest
                    break
                v = best_v
                d = dist[prev, v]
                bat -= d
                t += d / speed
                load += demands[v]
                if t < ready[v]:
                    t = ready[v]
                t += service[v]
                route.append(v)
                remaining.pop(best_idx)
                prev = v
            if route:
                routes.append(route)
            # if remaining, they'll be handled by greedy fallback below
            # store leftovers
            if remaining:
                # append as singleton routes for now (will be merged later)
                for v in remaining:
                    routes.append([v])
        return routes

    # ---- Greedy construction with a given customer order ----
    def greedy_construct(customer_order):
        remaining = list(customer_order)
        routes = []
        while remaining:
            route = []
            prev = 0
            t = 0.0
            bat = battery
            load = 0
            while remaining:
                best_v = None
                best_score = float('inf')
                best_idx = -1
                for idx, v in enumerate(remaining):
                    d = dist[prev, v]
                    bat_after = bat - d
                    t_after = t + d / speed
                    if bat_after < -1e-9:
                        continue
                    if load + demands[v] > capacity:
                        continue
                    if t_after < ready[v]:
                        t_after = ready[v]
                    if t_after > due[v] + 1e-9:
                        continue
                    slack = due[v] - t_after
                    score = d - 0.02 * slack
                    if score < best_score:
                        best_score = score
                        best_v = v
                        best_idx = idx
                if best_v is None:
                    break
                v = best_v
                d = dist[prev, v]
                bat -= d
                t += d / speed
                load += demands[v]
                if t < ready[v]:
                    t = ready[v]
                t += service[v]
                route.append(v)
                remaining.pop(best_idx)
                prev = v
            if route:
                routes.append(route)
        return routes

    # ---- Try K-means construction with several cluster counts / seeds ----
    if time.time() < deadline:
        for k in [4, 5, 6, 7]:
            if time.time() > deadline:
                break
            for seed in [0, 1, 2]:
                if time.time() > deadline:
                    break
                km_routes = kmeans_construct(k, seed)
                if validate_routes(km_routes):
                    d = total_route_distance(km_routes)
                    if d < best_distance:
                        best_routes = [list(r) for r in km_routes]
                        best_distance = d

    # ---- Try greedy construction with different orderings ----
    orderings = [
        sorted(customers, key=lambda v: due[v]),
        sorted(customers, key=lambda v: ready[v]),
        sorted(customers, key=lambda v: dist[0, v]),
        sorted(customers, key=lambda v: (due[v], dist[0, v])),
        sorted(customers, key=lambda v: (ready[v], dist[0, v])),
    ]
    for ordering in orderings:
        if time.time() > deadline:
            break
        greedy_routes = greedy_construct(ordering)
        if validate_routes(greedy_routes):
            d = total_route_distance(greedy_routes)
            if d < best_distance:
                best_routes = [list(r) for r in greedy_routes]
                best_distance = d

    # ---- 2-opt within a single route ----
    def two_opt_route(route):
        if len(route) < 2:
            return route
        improved = True
        while improved:
            improved = False
            n = len(route)
            for a in range(n - 1):
                if time.time() > deadline:
                    break
                for b in range(a + 1, n):
                    old_boundary = 0.0
                    if a > 0:
                        old_boundary += dist[route[a - 1], route[a]]
                    else:
                        old_boundary += dist[0, route[a]]
                    if b < n - 1:
                        old_boundary += dist[route[b], route[b + 1]]
                    else:
                        old_boundary += dist[route[b], 0]

                    new_boundary = 0.0
                    if a > 0:
                        new_boundary += dist[route[a - 1], route[b]]
                    else:
                        new_boundary += dist[0, route[b]]
                    if b < n - 1:
                        new_boundary += dist[route[a], route[b + 1]]
                    else:
                        new_boundary += dist[route[a], 0]

                    if new_boundary < old_boundary - 1e-6:
                        new_route = route[:a] + route[a:b + 1][::-1] + route[b + 1:]
                        feasible, _ = fast_check_route(new_route)
                        if feasible:
                            route = new_route
                            improved = True
                            break
                if improved:
                    break
        return route

    # ---- Or-opt: move a segment of length L (1..3) to a better position within a route ----
    def or_opt_route(route, L=3):
        if len(route) < 3:
            return route
        improved = True
        while improved:
            improved = False
            n = len(route)
            for length in range(1, L + 1):
                if time.time() > deadline:
                    break
                for i in range(n - length + 1):
                    if time.time() > deadline:
                        break
                    seg = route[i:i + length]
                    # candidate new positions (not overlapping original span)
                    # remove seg
                    base = route[:i] + route[i + length:]
                    # try inserting at each position in base
                    for j in range(len(base) + 1):
                        if j == i and j + length == i + length:
                            continue
                        new_route = base[:j] + seg + base[j:]
                        if new_route == route:
                            continue
                        nd = 0.0
                        prev = 0
                        for v in new_route:
                            nd += dist[prev, v]
                            prev = v
                        nd += dist[prev, 0]
                        # compare to current route distance
                        cd = 0.0
                        prev = 0
                        for v in route:
                            cd += dist[prev, v]
                            prev = v
                        cd += dist[prev, 0]
                        if nd < cd - 1e-6:
                            feasible, _ = fast_check_route(new_route)
                            if feasible:
                                route = new_route
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        return route

    # ---- Relocate (single node) local search across routes ----
    def relocate_search(routes, max_passes=8):
        routes = [list(r) for r in routes]
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            improved = False
            current_dist = total_route_distance(routes)
            for i in range(len(routes)):
                if time.time() > deadline:
                    break
                for j in range(len(routes[i])):
                    if time.time() > deadline:
                        break
                    v = routes[i][j]
                    route_i_without = routes[i][:j] + routes[i][j + 1:]
                    if not route_i_without:
                        new_routes = [r for idx, r in enumerate(routes) if idx != i]
                    else:
                        new_routes = list(routes)
                        new_routes[i] = route_i_without

                    if route_i_without:
                        feasible_i, _ = fast_check_route(route_i_without)
                        if not feasible_i:
                            continue
                    else:
                        feasible_i = True

                    for k in range(len(new_routes)):
                        if time.time() > deadline:
                            break
                        route_k = new_routes[k]
                        for pos in range(len(route_k) + 1):
                            new_route_k = route_k[:pos] + [v] + route_k[pos:]
                            feasible, _ = fast_check_route(new_route_k)
                            if not feasible:
                                continue
                            new_routes[k] = new_route_k
                            new_dist = total_route_distance(new_routes)
                            if new_dist < current_dist - 1e-6:
                                routes = new_routes
                                current_dist = new_dist
                                improved = True
                                break
                            else:
                                new_routes[k] = route_k
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return routes

    # ---- Cross-route swap local search ----
    def swap_search(routes):
        improved_swap = True
        while improved_swap and time.time() < deadline:
            improved_swap = False
            current_dist = total_route_distance(routes)
            for i in range(len(routes)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(routes)):
                    if time.time() > deadline:
                        continue
                    for a in range(len(routes[i])):
                        if time.time() > deadline:
                            break
                        for b in range(len(routes[j])):
                            vi = routes[i][a]
                            vj = routes[j][b]
                            new_routes = []
                            for r in range(len(routes)):
                                if r == i:
                                    nr = routes[i][:]
                                    nr[a] = vj
                                    new_routes.append(nr)
                                elif r == j:
                                    nr = routes[j][:]
                                    nr[b] = vi
                                    new_routes.append(nr)
                                else:
                                    new_routes.append(routes[r][:])
                            feasible = True
                            for route in new_routes:
                                f, _ = fast_check_route(route)
                                if not f:
                                    feasible = False
                                    break
                            if feasible:
                                new_dist = total_route_distance(new_routes)
                                if new_dist < current_dist - 1e-6:
                                    routes = new_routes
                                    current_dist = new_dist
                                    improved_swap = True
                                    break
                        if improved_swap:
                            break
                    if improved_swap:
                        break
                if improved_swap:
                    break
        return routes

    # ---- Refinement loop: 2-opt + Or-opt + relocate + swap, iterate while improving ----
    if time.time() < deadline:
        for _ in range(6):
            if time.time() > deadline:
                break
            # intra-route 2-opt
            for i in range(len(best_routes)):
                if time.time() > deadline:
                    break
                best_routes[i] = two_opt_route(best_routes[i])
            # intra-route Or-opt (1-3 segment moves)
            for i in range(len(best_routes)):
                if time.time() > deadline:
                    break
                best_routes[i] = or_opt_route(best_routes[i], L=3)
            # relocate across routes
            if time.time() < deadline:
                best_routes = relocate_search(best_routes, max_passes=4)
            # cross-route swap
            if time.time() < deadline:
                best_routes = swap_search(best_routes)
            # remove empty routes
            best_routes = [r for r in best_routes if r]

    # ---- Final high-precision polish pass (tighter tolerance, more iterations) ----
    if time.time() < deadline:
        for i in range(len(best_routes)):
            if time.time() > deadline:
                break
            best_routes[i] = two_opt_route(best_routes[i])
        if time.time() < deadline:
            for i in range(len(best_routes)):
                if time.time() > deadline:
                    break
                best_routes[i] = or_opt_route(best_routes[i], L=3)
        if time.time() < deadline:
            best_routes = relocate_search(best_routes, max_passes=3)
        if time.time() < deadline:
            for i in range(len(best_routes)):
                if time.time() > deadline:
                    break
                best_routes[i] = two_opt_route(best_routes[i])

    # Remove empty routes
    best_routes = [r for r in best_routes if r]

    return best_routes
# EVOLVE-BLOCK-END
