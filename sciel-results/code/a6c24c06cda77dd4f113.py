# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def plan(instance):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit

    coords = np.array(instance["coords"], dtype=float)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    # Precompute distance matrix
    n_nodes = len(coords)
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            dist[i, j] = np.sqrt((coords[i, 0] - coords[j, 0])**2 + (coords[i, 1] - coords[j, 1])**2)

    customers = list(range(1, n_customers + 1))

    def simulate_route(route, add_station_at=None, add_station_before=None):
        """
        Simulate a route (list of customer node ids) starting from depot (node 0).
        Returns (total_distance, feasible, final_time, final_battery) or None if infeasible.
        If add_station_at is not None, insert a station visit at that position.
        If add_station_before is not None, insert a station visit before that index.
        """
        if add_station_at is not None:
            # Build route with station inserted
            full_route = list(route[:add_station_at]) + [add_station_at] + list(route[add_station_at:])
        elif add_station_before is not None:
            full_route = list(route[:add_station_before]) + [add_station_before] + list(route[add_station_before:])
        else:
            full_route = list(route)

        t = 0.0
        battery = battery_cap
        total_dist = 0.0
        current = 0  # depot

        for node in full_route:
            d = dist[current, node]
            t += d / speed
            battery -= d
            total_dist += d

            if battery < -1e-9:
                return None  # battery went negative

            if node in customers:
                # Customer service
                if t < ready[node]:
                    t = ready[node]
                if t > due[node] + 1e-9:
                    return None  # missed due time
                t += service[node]
            elif node in stations:
                # Charging station
                # Recharge time: (battery_cap - battery) * recharge_rate
                recharge_time = (battery_cap - battery) * recharge_rate
                t += recharge_time
                battery = battery_cap

            current = node

        # Return to depot
        d = dist[current, 0]
        t += d / speed
        battery -= d
        total_dist += d

        if battery < -1e-9:
            return None
        if t > horizon + 1e-9:
            return None

        return (total_dist, True, t, battery)

    def route_demand(route):
        return sum(demands[v] for v in route)

    def try_build_routes(greedy_order):
        """
        Given an ordering of customers, greedily pack them into routes.
        Returns list of routes or None if infeasible.
        """
        unvisited = set(customers)
        routes = []

        for c in greedy_order:
            if c not in unvisited:
                continue

            # Try to add c to existing routes
            placed = False
            for i, route in enumerate(routes):
                new_route = route + [c]
                if route_demand(new_route) > capacity:
                    continue
                sim = simulate_route(new_route)
                if sim is not None:
                    routes[i] = new_route
                    unvisited.remove(c)
                    placed = True
                    break

            if not placed:
                # Try to add c to existing routes with a station inserted
                for i, route in enumerate(routes):
                    if route_demand(route + [c]) > capacity:
                        continue
                    # Try inserting station at various positions
                    for pos in range(len(route) + 1):
                        # Try inserting a station before position pos
                        for st in stations:
                            new_route_with_st = list(route[:pos]) + [st] + list(route[pos:]) + [c]
                            # Check demand (station doesn't add demand)
                            if route_demand(new_route_with_st) > capacity:
                                continue
                            sim = simulate_route(new_route_with_st)
                            if sim is not None:
                                routes[i] = new_route_with_st
                                unvisited.remove(c)
                                placed = True
                                break
                        if placed:
                            break
                    if placed:
                        break

            if not placed:
                # Start a new route
                new_route = [c]
                sim = simulate_route(new_route)
                if sim is None:
                    return None  # Can't even serve c alone
                routes.append(new_route)
                unvisited.remove(c)

        if unvisited:
            return None

        return routes

    def total_distance(routes):
        total = 0.0
        for route in routes:
            total += simulate_route(route)[0]
        return total

    def two_opt(route, max_iter=100):
        """
        Apply 2-opt improvement to a single route, maintaining feasibility.
        """
        improved = True
        iterations = 0
        while improved and iterations < max_iter:
            improved = False
            iterations += 1
            n = len(route)
            for i in range(n - 1):
                for j in range(i + 1, n):
                    # Reverse segment i..j
                    new_route = list(route[:i]) + list(reversed(route[i:j+1])) + list(route[j+1:])
                    if route_demand(new_route) > capacity:
                        continue
                    sim_old = simulate_route(route)
                    sim_new = simulate_route(new_route)
                    if sim_old is None or sim_new is None:
                        continue
                    if sim_new[0] < sim_old[0] - 1e-6:
                        route = new_route
                        improved = True

        return route

    def two_opt_all_routes(routes, max_iter_per_route=50):
        for i in range(len(routes)):
            routes[i] = two_opt(routes[i], max_iter_per_route)
        return routes

    def relocate_move(routes):
        """
        Try to move a customer from one route to another, or to a new route,
        if it improves total distance and maintains feasibility.
        """
        improved = False
        for i, route_i in enumerate(routes):
            for pos_i, c in enumerate(route_i):
                # Try moving c to each other route
                for j, route_j in enumerate(routes):
                    if i == j:
                        continue
                    if route_demand(route_j + [c]) > capacity:
                        continue
                    # Remove c from route_i
                    new_route_i = list(route_i[:pos_i]) + list(route_i[pos_i+1:])
                    # Try inserting c at various positions in route_j
                    for pos_j in range(len(route_j) + 1):
                        new_route_j = list(route_j[:pos_j]) + [c] + list(route_j[pos_j:])
                        sim_i = simulate_route(new_route_i) if new_route_i else (0.0, True, 0.0, battery_cap)
                        sim_j = simulate_route(new_route_j)
                        if sim_i is None or sim_j is None:
                            continue
                        # Check if this improves total distance
                        old_dist = simulate_route(route_i)[0] + simulate_route(route_j)[0]
                        new_dist = sim_i[0] + sim_j[0]
                        if new_dist < old_dist - 1e-6:
                            routes[i] = new_route_i
                            routes[j] = new_route_j
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        # Remove empty routes
        routes = [r for r in routes if r]
        return routes, improved

    def swap_move(routes):
        """
        Try swapping two customers between routes if it improves total distance.
        """
        improved = False
        for i, route_i in enumerate(routes):
            for j, route_j in enumerate(routes):
                if i >= j:
                    continue
                for pos_i, c_i in enumerate(route_i):
                    for pos_j, c_j in enumerate(route_j):
                        if demands[c_i] == demands[c_j]:
                            continue  # Swap doesn't change demand
                        # Swap c_i and c_j
                        new_route_i = list(route_i)
                        new_route_j = list(route_j)
                        new_route_i[pos_i] = c_j
                        new_route_j[pos_j] = c_i
                        if route_demand(new_route_i) > capacity or route_demand(new_route_j) > capacity:
                            continue
                        sim_i = simulate_route(new_route_i)
                        sim_j = simulate_route(new_route_j)
                        if sim_i is None or sim_j is None:
                            continue
                        old_dist = simulate_route(route_i)[0] + simulate_route(route_j)[0]
                        new_dist = sim_i[0] + sim_j[0]
                        if new_dist < old_dist - 1e-6:
                            routes[i] = new_route_i
                            routes[j] = new_route_j
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        return routes, improved

    # Initial construction: sort by due time
    initial_order = sorted(customers, key=lambda v: due[v])
    routes = try_build_routes(initial_order)
    if routes is None:
        # Fallback: one customer per route
        routes = [[c] for c in customers]

    # Local search
    best_routes = routes
    best_dist = total_distance(best_routes)

    # 2-opt within routes
    best_routes = two_opt_all_routes(best_routes, 30)
    best_dist = total_distance(best_routes)

    # Relocate and swap moves
    for iteration in range(200):
        if time.time() > deadline - 5.0:
            break

        routes_copy = [list(r) for r in best_routes]
        routes_copy, imp1 = relocate_move(routes_copy)
        if imp1:
            d = total_distance(routes_copy)
            if d < best_dist - 1e-6:
                best_routes = routes_copy
                best_dist = d
                # 2-opt after relocate
                best_routes = two_opt_all_routes(best_routes, 10)
                best_dist = total_distance(best_routes)
                continue

        routes_copy = [list(r) for r in best_routes]
        routes_copy, imp2 = swap_move(routes_copy)
        if imp2:
            d = total_distance(routes_copy)
            if d < best_dist - 1e-6:
                best_routes = routes_copy
                best_dist = d
                continue

        # Try 2-opt again
        routes_copy = two_opt_all_routes([list(r) for r in best_routes], 20)
        d = total_distance(routes_copy)
        if d < best_dist - 1e-6:
            best_routes = routes_copy
            best_dist = d
        else:
            # No improvement, try a few more iterations
            pass

    # Try different initial orderings if time permits
    if time.time() < deadline - 10.0:
        for _ in range(5):
            random_order = customers[:]
            random.shuffle(random_order)
            routes2 = try_build_routes(random_order)
            if routes2 is not None:
                routes2 = two_opt_all_routes(routes2, 20)
                d2 = total_distance(routes2)
                if d2 < best_dist - 1e-6:
                    best_routes = routes2
                    best_dist = d2

    # Final check: ensure all customers are served exactly once
    all_served = []
    for route in best_routes:
        for c in route:
            if c in customers:
                all_served.append(c)

    # If something went wrong, fall back to one-per-route
    if len(all_served) != len(customers) or len(set(all_served)) != len(customers):
        best_routes = [[c] for c in customers]

    return best_routes
# EVOLVE-BLOCK-END
