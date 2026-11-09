"""Improved EV routing with time windows and charging stations."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def plan(instance):
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s kill

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

    def check_route(route):
        """Check if a route is feasible. Returns (feasible, total_distance)."""
        if not route:
            return True, 0.0

        # Check capacity
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
            t += d / speed
            total_dist += d

            if bat < -1e-9:
                return False, 0.0

            # At customer
            if t < ready[v]:
                t = ready[v]
            if t > due[v] + 1e-9:
                return False, 0.0
            t += service[v]
            prev = v

        # Return to depot
        d = dist[prev, 0]
        bat -= d
        t += d / speed
        total_dist += d

        if bat < -1e-9:
            return False, 0.0
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

    def greedy_construct(customer_order):
        """Construct routes using greedy nearest neighbor with time windows."""
        remaining = list(customer_order)
        routes = []

        while remaining:
            route = []
            prev = 0
            t = 0.0
            bat = battery
            load = 0

            while remaining:
                best_customer = None
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

                    # Check time window
                    if t_after < ready[v]:
                        t_after = ready[v]
                    if t_after > due[v] + 1e-9:
                        continue

                    # Score: distance + small time penalty
                    score = d + (t_after - ready[v]) * 0.05

                    if score < best_score:
                        best_score = score
                        best_customer = v
                        best_idx = idx

                if best_customer is None:
                    break

                v = best_customer
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

    def validate_routes(routes):
        """Check if routes are valid: all customers served exactly once, all feasible."""
        all_customers = []
        for route in routes:
            feasible, _ = check_route(route)
            if not feasible:
                return False
            all_customers.extend(route)
        if len(all_customers) != len(set(all_customers)):
            return False
        if set(all_customers) != set(customers):
            return False
        return True

    def two_opt_route(route, max_iter=100):
        """Apply 2-opt to a single route until no improvement."""
        if len(route) < 2:
            return route
        improved = True
        n = len(route)
        iter_count = 0
        while improved and iter_count < max_iter and time.time() < deadline:
            improved = False
            iter_count += 1
            n = len(route)
            for a in range(n - 1):
                if time.time() > deadline:
                    break
                for b in range(a + 1, n):
                    # Compute boundary distances
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
                        feasible, _ = check_route(new_route)
                        if feasible:
                            route = new_route
                            improved = True
                            break
                if improved:
                    break
        return route

    def or_opt_route(route, max_iter=50):
        """Apply or-opt (move single node) to a single route."""
        if len(route) < 3:
            return route
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter and time.time() < deadline:
            improved = False
            iter_count += 1
            n = len(route)
            for i in range(n):
                if time.time() > deadline:
                    break
                v = route[i]
                # Remove v
                route_without = route[:i] + route[i+1:]
                if not route_without:
                    continue

                # Try inserting v at each position
                best_pos = -1
                best_delta = 0.0

                # Calculate old cost contribution of removing v
                old_cost = 0.0
                if i > 0:
                    old_cost += dist[route[i-1], v]
                else:
                    old_cost += dist[0, v]
                if i < n - 1:
                    old_cost += dist[v, route[i+1]]
                else:
                    old_cost += dist[v, 0]
                # New cost after removing v
                if i > 0 and i < n - 1:
                    old_cost -= dist[route[i-1], route[i+1]]
                elif i == 0 and n > 1:
                    old_cost -= dist[0, route[1]]
                elif i == n - 1 and n > 1:
                    old_cost -= dist[route[n-2], 0]

                for pos in range(len(route_without) + 1):
                    # Calculate new cost of inserting v at pos
                    new_cost = 0.0
                    if pos > 0:
                        new_cost += dist[route_without[pos-1], v]
                    else:
                        new_cost += dist[0, v]
                    if pos < len(route_without):
                        new_cost += dist[v, route_without[pos]]
                    else:
                        new_cost += dist[v, 0]
                    # Subtract the edge that would be broken
                    if pos > 0 and pos < len(route_without):
                        new_cost -= dist[route_without[pos-1], route_without[pos]]
                    elif pos == 0 and len(route_without) > 0:
                        new_cost -= dist[0, route_without[0]]
                    elif pos == len(route_without) and len(route_without) > 0:
                        new_cost -= dist[route_without[-1], 0]

                    delta = new_cost - old_cost
                    if delta < best_delta - 1e-6:
                        best_delta = delta
                        best_pos = pos

                if best_pos >= 0 and best_delta < -1e-6:
                    # Perform the move
                    new_route = route_without[:best_pos] + [v] + route_without[best_pos:]
                    feasible, _ = check_route(new_route)
                    if feasible:
                        route = new_route
                        improved = True
                        break
        return route

    def relocate_search(routes, max_passes=10):
        """Remove-and-reinsert local search."""
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
                    # Remove v from route i
                    route_i_without = routes[i][:j] + routes[i][j+1:]
                    if not route_i_without:
                        new_routes = [r for idx, r in enumerate(routes) if idx != i]
                    else:
                        new_routes = list(routes)
                        new_routes[i] = route_i_without

                    # Check feasibility of modified route i
                    if route_i_without:
                        feasible_i, _ = check_route(route_i_without)
                        if not feasible_i:
                            continue
                    else:
                        feasible_i = True

                    # Try inserting v into each position of each other route
                    for k in range(len(new_routes)):
                        if time.time() > deadline:
                            break
                        route_k = new_routes[k]
                        for pos in range(len(route_k) + 1):
                            new_route_k = route_k[:pos] + [v] + route_k[pos:]
                            feasible, _ = check_route(new_route_k)
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

    def swap_search(routes, max_passes=5):
        """Cross-route swap local search."""
        routes = [list(r) for r in routes]
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            improved = False
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
                            # Swap
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
                                f, _ = check_route(route)
                                if not f:
                                    feasible = False
                                    break

                            if feasible:
                                new_dist = total_route_distance(new_routes)
                                if new_dist < current_dist - 1e-6:
                                    routes = new_routes
                                    current_dist = new_dist
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
        return routes

    # Generate multiple initial solutions with different orderings
    orderings = [
        sorted(customers, key=lambda v: due[v]),
        sorted(customers, key=lambda v: ready[v]),
        sorted(customers, key=lambda v: dist[0, v]),
        sorted(customers, key=lambda v: (due[v], dist[0, v])),
        sorted(customers, key=lambda v: (ready[v], dist[0, v])),
        sorted(customers, key=lambda v: (due[v] - ready[v], dist[0, v])),
    ]

    # Also try random orderings
    for _ in range(5):
        random.shuffle(customers)
        orderings.append(customers[:])

    best_routes = None
    best_distance = float('inf')

    for ordering in orderings:
        if time.time() > deadline:
            break
        greedy_routes = greedy_construct(ordering)
        if validate_routes(greedy_routes):
            d = total_route_distance(greedy_routes)
            if d < best_distance:
                best_routes = [list(r) for r in greedy_routes]
                best_distance = d

    if best_routes is None:
        # Fallback: one customer per route
        best_routes = [[v] for v in customers]
        best_distance = sum(dist[0, v] * 2 for v in customers)

    # Apply local search improvements
    # 1. 2-opt on each route
    if time.time() < deadline:
        for i in range(len(best_routes)):
            if time.time() > deadline:
                break
            best_routes[i] = two_opt_route(best_routes[i])

    # 2. Or-opt on each route
    if time.time() < deadline:
        for i in range(len(best_routes)):
            if time.time() > deadline:
                break
            best_routes[i] = or_opt_route(best_routes[i])

    # 3. Relocate search
    if time.time() < deadline:
        best_routes = relocate_search(best_routes, max_passes=8)

    # 4. Swap search
    if time.time() < deadline:
        best_routes = swap_search(best_routes, max_passes=4)

    # 5. Another round of 2-opt and or-opt
    if time.time() < deadline:
        for i in range(len(best_routes)):
            if time.time() > deadline:
                break
            best_routes[i] = two_opt_route(best_routes[i], max_iter=50)

    if time.time() < deadline:
        for i in range(len(best_routes)):
            if time.time() > deadline:
                break
            best_routes[i] = or_opt_route(best_routes[i], max_iter=30)

    # 6. Final relocate
    if time.time() < deadline:
        best_routes = relocate_search(best_routes, max_passes=5)

    # 7. Final 2-opt polish
    if time.time() < deadline:
        for i in range(len(best_routes)):
            if time.time() > deadline:
                break
            best_routes[i] = two_opt_route(best_routes[i], max_iter=30)

    # Remove empty routes
    best_routes = [r for r in best_routes if r]

    return best_routes
# EVOLVE-BLOCK-END
