"""Rewritten program for EV routing with time windows."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def plan(instance):
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

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

    # Precompute distance matrix
    n_nodes = len(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    customers = list(range(1, n_customers + 1))
    station_set = set(stations)

    def check_route_feasibility(route):
        """Check if a route is feasible. Returns (feasible, total_distance)."""
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

            if v in station_set:
                # Recharge: time added is (current_battery) * recharge_rate
                # Actually, the problem says: station adds (battery - energy) * recharge_rate time
                # Wait, let me re-read: "a station adds (battery - energy) * recharge_rate time and refills"
                # I think "energy" here means the current battery level. So it's (battery - current_battery) * recharge_rate?
                # No, that doesn't make sense. Let me re-read carefully.
                # "a station adds (battery - energy) * recharge_rate time and refills"
                # I think this means the time to recharge is proportional to how much energy is missing.
                # So time = (max_battery - current_battery) * recharge_rate
                # And after that, battery is refilled to max.
                recharge_time = (battery - bat) * recharge_rate
                t += recharge_time
                bat = battery
            else:
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
            feasible, _ = check_route_feasibility(route)
            if not feasible:
                return False
            all_customers.extend(route)
        if len(all_customers) != len(set(all_customers)):
            return False
        if set(all_customers) != set(customers):
            return False
        return True

    # Initial solution: one customer per route
    best_routes = [[v] for v in customers]
    best_distance = sum(dist[0, v] * 2 for v in customers)

    # --- Construction: Savings with feasibility ---
    def savings_construct():
        """Clarke-Wright savings based construction with feasibility checks."""
        routes = [[v] for v in customers]
        route_of = {v: i for i, v in enumerate(customers)}

        savings = []
        for i in range(n_customers):
            for j in range(i + 1, n_customers):
                vi, vj = i + 1, j + 1
                s = dist[0, vi] + dist[0, vj] - dist[vi, vj]
                if s > 0:
                    savings.append((s, vi, vj))

        savings.sort(reverse=True)

        for s, vi, vj in savings:
            if time.time() > deadline:
                break
            ri = route_of.get(vi)
            rj = route_of.get(vj)
            if ri is None or rj is None or ri == rj:
                continue

            route_i = routes[ri]
            route_j = routes[rj]

            can_merge = False
            new_route = None

            # Try all merge options where vi and vj are at the ends
            options = []
            if route_i[-1] == vi and route_j[0] == vj:
                options.append(route_i + route_j)
            if route_j[-1] == vj and route_i[0] == vi:
                options.append(route_j + route_i)
            if route_i[0] == vi and route_j[0] == vj:
                options.append(route_i[::-1] + route_j)
            if route_i[-1] == vi and route_j[-1] == vj:
                options.append(route_i + route_j[::-1])
            if route_i[0] == vi and route_j[-1] == vj:
                options.append(route_i[::-1] + route_j[::-1])
            if route_j[0] == vj and route_i[-1] == vi:
                options.append(route_j + route_i[::-1])

            for candidate in options:
                if sum(demands[x] for x in candidate) > capacity:
                    continue
                f, _ = check_route_feasibility(candidate)
                if f:
                    can_merge = True
                    new_route = candidate
                    break

            if can_merge:
                routes[ri] = new_route
                routes[rj] = []
                non_empty = [r for r in routes if r]
                route_of = {}
                for i, r in enumerate(non_empty):
                    for v in r:
                        route_of[v] = i
                routes = non_empty

        return [r for r in routes if r]

    # --- Construction: Greedy insertion ---
    def greedy_insertion(ordering):
        """Insert customers one by one in the given order, best feasible position."""
        routes = []
        remaining = list(ordering)

        while remaining:
            v = remaining.pop(0)
            best_route_idx = -1
            best_pos = -1
            best_cost = float('inf')

            # Try inserting into existing routes
            for ri, route in enumerate(routes):
                for pos in range(len(route) + 1):
                    new_route = route[:pos] + [v] + route[pos:]
                    if sum(demands[x] for x in new_route) > capacity:
                        continue
                    f, _ = check_route_feasibility(new_route)
                    if not f:
                        continue
                    # Cost: increase in total distance
                    old_d = 0.0
                    if pos == 0:
                        old_d = dist[0, route[0]]
                    elif pos == len(route):
                        old_d = dist[route[-1], 0]
                    else:
                        old_d = dist[route[pos-1], route[pos]]
                    
                    new_d = 0.0
                    if pos == 0:
                        new_d = dist[0, v] + dist[v, route[0]]
                    elif pos == len(route):
                        new_d = dist[route[-1], v] + dist[v, 0]
                    else:
                        new_d = dist[route[pos-1], v] + dist[v, route[pos]]
                    
                    cost = new_d - old_d
                    if cost < best_cost:
                        best_cost = cost
                        best_route_idx = ri
                        best_pos = pos

            # If no feasible insertion, start new route
            if best_route_idx == -1:
                routes.append([v])
            else:
                routes[best_route_idx] = routes[best_route_idx][:best_pos] + [v] + routes[best_route_idx][best_pos:]

        return routes

    # Try savings construction
    if time.time() < deadline:
        sw_routes = savings_construct()
        if validate_routes(sw_routes):
            d = total_route_distance(sw_routes)
            if d < best_distance:
                best_routes = sw_routes
                best_distance = d

    # Try greedy insertion with different orderings
    orderings = [
        sorted(customers, key=lambda v: due[v]),
        sorted(customers, key=lambda v: ready[v]),
        sorted(customers, key=lambda v: dist[0, v]),
        sorted(customers, key=lambda v: (due[v], dist[0, v])),
        sorted(customers, key=lambda v: (ready[v], dist[0, v])),
        sorted(customers, key=lambda v: (dist[0, v], due[v])),
    ]

    for ordering in orderings:
        if time.time() > deadline:
            break
        gi_routes = greedy_insertion(ordering)
        if validate_routes(gi_routes):
            d = total_route_distance(gi_routes)
            if d < best_distance:
                best_routes = gi_routes
                best_distance = d

    # --- Local Search: 2-opt ---
    def two_opt_route(route):
        """Apply 2-opt to a single route until no improvement."""
        if len(route) < 2:
            return route
        improved = True
        while improved:
            improved = False
            n = len(route)
            for a in range(n - 1):
                if time.time() > deadline:
                    return route
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
                        feasible, _ = check_route_feasibility(new_route)
                        if feasible:
                            route = new_route
                            improved = True
                            break
                if improved:
                    break
        return route

    # Apply 2-opt to all routes
    if time.time() < deadline:
        for i in range(len(best_routes)):
            if time.time() > deadline:
                break
            best_routes[i] = two_opt_route(best_routes[i])

    # --- Local Search: Relocate (remove and reinsert) ---
    def relocate_search(routes, max_passes=8):
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
                    route_i_without = routes[i][:j] + routes[i][j+1:]
                    if not route_i_without:
                        new_routes = [r for idx, r in enumerate(routes) if idx != i]
                    else:
                        new_routes = list(routes)
                        new_routes[i] = route_i_without

                    if route_i_without:
                        feasible_i, _ = check_route_feasibility(route_i_without)
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
                            feasible, _ = check_route_feasibility(new_route_k)
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

    # Apply relocate search
    if time.time() < deadline:
        best_routes = relocate_search(best_routes, max_passes=5)

    # --- Local Search: Cross-route swap ---
    if time.time() < deadline:
        improved_swap = True
        while improved_swap and time.time() < deadline:
            improved_swap = False
            current_dist = total_route_distance(best_routes)
            for i in range(len(best_routes)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(best_routes)):
                    if time.time() > deadline:
                        continue
                    for a in range(len(best_routes[i])):
                        if time.time() > deadline:
                            break
                        for b in range(len(best_routes[j])):
                            vi = best_routes[i][a]
                            vj = best_routes[j][b]
                            new_routes = []
                            for r in range(len(best_routes)):
                                if r == i:
                                    nr = best_routes[i][:]
                                    nr[a] = vj
                                    new_routes.append(nr)
                                elif r == j:
                                    nr = best_routes[j][:]
                                    nr[b] = vi
                                    new_routes.append(nr)
                                else:
                                    new_routes.append(best_routes[r][:])

                            feasible = True
                            for route in new_routes:
                                f, _ = check_route_feasibility(route)
                                if not f:
                                    feasible = False
                                    break

                            if feasible:
                                new_dist = total_route_distance(new_routes)
                                if new_dist < current_dist - 1e-6:
                                    best_routes = new_routes
                                    current_dist = new_dist
                                    improved_swap = True
                                    break
                        if improved_swap:
                            break
                    if improved_swap:
                        break
                if improved_swap:
                    break

    # --- Large Neighborhood Search (LNS) ---
    def lns_search(routes, max_iterations=50):
        """LNS: destroy a random subset, reconstruct via greedy insertion."""
        routes = [list(r) for r in routes]
        current_dist = total_route_distance(routes)
        best_local = list(routes)
        best_local_dist = current_dist
        temperature = 0.5  # initial acceptance probability scale

        for it in range(max_iterations):
            if time.time() > deadline:
                break

            # Destroy: pick a random subset of customers
            all_nodes = [v for r in routes for v in r]
            subset_size = max(2, min(len(all_nodes) // 3, 8))
            destroyed = random.sample(all_nodes, subset_size)
            destroyed_set = set(destroyed)

            # Remove destroyed customers from routes
            remaining_routes = []
            for r in routes:
                new_r = [v for v in r if v not in destroyed_set]
                if new_r:
                    remaining_routes.append(new_r)

            # Reconstruct: insert destroyed customers one by one
            # Sort by due date for better feasibility
            destroyed_sorted = sorted(destroyed, key=lambda v: due[v])

            new_routes = [list(r) for r in remaining_routes]

            for v in destroyed_sorted:
                best_route_idx = -1
                best_pos = -1
                best_cost = float('inf')

                for ri, route in enumerate(new_routes):
                    for pos in range(len(route) + 1):
                        new_route = route[:pos] + [v] + route[pos:]
                        if sum(demands[x] for x in new_route) > capacity:
                            continue
                        f, _ = check_route_feasibility(new_route)
                        if not f:
                            continue

                        # Cost: increase in total distance
                        old_d = 0.0
                        if pos == 0:
                            old_d = dist[0, route[0]]
                        elif pos == len(route):
                            old_d = dist[route[-1], 0]
                        else:
                            old_d = dist[route[pos-1], route[pos]]

                        new_d = 0.0
                        if pos == 0:
                            new_d = dist[0, v] + dist[v, route[0]]
                        elif pos == len(route):
                            new_d = dist[route[-1], v] + dist[v, 0]
                        else:
                            new_d = dist[route[pos-1], v] + dist[v, route[pos]]

                        cost = new_d - old_d
                        if cost < best_cost:
                            best_cost = cost
                            best_route_idx = ri
                            best_pos = pos

                if best_route_idx == -1:
                    # No feasible insertion found - this reconstruction failed
                    break
                else:
                    new_routes[best_route_idx] = new_routes[best_route_idx][:best_pos] + [v] + new_routes[best_route_idx][best_pos:]

            # Check if reconstruction succeeded
            if len([v for r in new_routes for v in r]) != len(customers):
                continue

            # Validate
            if not validate_routes(new_routes):
                continue

            new_dist = total_route_distance(new_routes)

            # Accept if better, or with some probability if worse (simulated annealing style)
            if new_dist < current_dist - 1e-6:
                routes = new_routes
                current_dist = new_dist
                if current_dist < best_local_dist:
                    best_local = [list(r) for r in routes]
                    best_local_dist = current_dist
            else:
                # Accept worse solutions with probability exp(-delta/T)
                delta = new_dist - current_dist
                if delta < 50:  # only accept modestly worse solutions
                    prob = math.exp(-delta / temperature)
                    if random.random() < prob:
                        routes = new_routes
                        current_dist = new_dist

            # Cool down
            temperature *= 0.95

        return best_local

    # Apply LNS
    if time.time() < deadline:
        best_routes = lns_search(best_routes, max_iterations=40)

    # Final polish: 2-opt + relocate
    if time.time() < deadline:
        for i in range(len(best_routes)):
            if time.time() > deadline:
                break
            best_routes[i] = two_opt_route(best_routes[i])

        if time.time() < deadline:
            best_routes = relocate_search(best_routes, max_passes=3)

        # Final 2-opt
        if time.time() < deadline:
            for i in range(len(best_routes)):
                if time.time() > deadline:
                    break
                best_routes[i] = two_opt_route(best_routes[i])

    # Remove empty routes
    best_routes = [r for r in best_routes if r]

    return best_routes
# EVOLVE-BLOCK-END
