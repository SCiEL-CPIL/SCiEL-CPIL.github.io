"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin

    coords = np.array(instance["coords"], dtype=float)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    # Precompute distance matrix
    n_nodes = len(coords)
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
            dist[i][j] = d
            dist[j][i] = d

    customers = list(range(1, n_customers + 1))

    def simulate_route(route):
        """Simulate a route starting from depot. Returns (total_dist, feasible, end_time)."""
        if not route:
            return 0.0, True, 0.0

        t = 0.0
        b = battery
        total_d = 0.0
        prev = 0

        for node in route:
            d = dist[prev][node]
            t += d / speed
            b -= d
            total_d += d

            if b < -1e-9:
                return total_d, False, t

            if node in stations:
                # Recharge time
                if b < battery:
                    t += (battery - b) * recharge_rate
                b = battery
            else:
                # Customer
                if t < ready[node]:
                    t = ready[node]
                if t > due[node] + 1e-9:
                    return total_d, False, t
                t += service[node]

            prev = node

        # Return to depot
        d = dist[prev][0]
        t += d / speed
        b -= d
        total_d += d

        if b < -1e-9:
            return total_d, False, t
        if t > horizon + 1e-9:
            return total_d, False, t

        return total_d, True, t

    def total_demand(route):
        return sum(demands[v] for v in route if v not in stations)

    # Greedy construction: cluster customers by proximity, build routes
    def construct_greedy():
        unvisited = set(customers)
        routes = []

        # Sort customers by due time for tie-breaking
        cust_sorted = sorted(customers, key=lambda v: (due[v], dist[0][v]))

        while unvisited:
            # Start new route from depot
            route = []
            current_node = 0
            t = 0.0
            b = battery
            total_d = 0.0
            load = 0

            while unvisited:
                # Find nearest unvisited customer that can be added feasibly
                best_cust = None
                best_dist = float('inf')

                for c in cust_sorted:
                    if c not in unvisited:
                        continue
                    if load + demands[c] > capacity:
                        continue

                    # Check if we can reach c and serve it
                    d_to_c = dist[current_node][c]
                    new_b = b - d_to_c
                    if new_b < -1e-9:
                        continue

                    new_t = t + d_to_c / speed
                    if new_t < ready[c]:
                        new_t = ready[c]
                    if new_t > due[c] + 1e-9:
                        continue
                    new_t += service[c]

                    # Check if we can return to depot after serving c
                    d_back = dist[c][0]
                    final_b = new_b - d_back
                    final_t = new_t + d_back / speed
                    if final_b < -1e-9 or final_t > horizon + 1e-9:
                        continue

                    if d_to_c < best_dist:
                        best_dist = d_to_c
                        best_cust = c

                if best_cust is None:
                    break

                # Add customer to route
                d_to_c = dist[current_node][best_cust]
                t += d_to_c / speed
                b -= d_to_c
                total_d += d_to_c

                if t < ready[best_cust]:
                    t = ready[best_cust]
                t += service[best_cust]

                load += demands[best_cust]
                route.append(best_cust)
                current_node = best_cust
                unvisited.discard(best_cust)

            if route:
                # Check full route feasibility
                _, feasible, _ = simulate_route(route)
                if feasible:
                    routes.append(route)
                else:
                    # Fallback: serve one by one
                    for c in route:
                        routes.append([c])
                    unvisited.update(route)

        return routes

    # 2-opt improvement for a single route
    def two_opt_route(route):
        """Apply 2-opt to a route, maintaining feasibility."""
        if len(route) < 2:
            return route

        improved = True
        best_route = list(route)
        best_dist, _, _ = simulate_route(best_route)

        while improved and (time.time() < deadline - 5.0):
            improved = False
            for i in range(len(best_route) - 1):
                if time.time() >= deadline - 5.0:
                    break
                for j in range(i + 1, len(best_route)):
                    # Reverse segment [i+1..j]
                    new_route = best_route[:i+1] + best_route[i+1:j+1][::-1] + best_route[j+1:]
                    _, feasible, _ = simulate_route(new_route)
                    if feasible:
                        nd, _, _ = simulate_route(new_route)
                        if nd < best_dist - 1e-9:
                            best_route = new_route
                            best_dist = nd
                            improved = True
                            break
                if improved:
                    break

        return best_route

    # Build initial solution
    routes = construct_greedy()

    # Apply 2-opt to each route
    for i in range(len(routes)):
        routes[i] = two_opt_route(routes[i])

    # Try to merge routes to reduce total distance
    def try_merge_routes(routes):
        """Try to merge pairs of routes if feasible."""
        merged = True
        while merged and (time.time() < deadline - 5.0):
            merged = False
            new_routes = []
            used = [False] * len(routes)

            for i in range(len(routes)):
                if used[i]:
                    continue
                for j in range(i + 1, len(routes)):
                    if used[j]:
                        continue
                    # Try concatenating routes[i] + routes[j]
                    combined = routes[i] + routes[j]
                    if total_demand(combined) > capacity:
                        continue
                    _, feasible, _ = simulate_route(combined)
                    if feasible:
                        new_routes.append(combined)
                        used[i] = True
                        used[j] = True
                        merged = True
                        break

            if merged:
                for k in range(len(routes)):
                    if not used[k]:
                        new_routes.append(routes[k])
                routes = new_routes

        return routes

    routes = try_merge_routes(routes)

    # Final 2-opt pass on merged routes
    for i in range(len(routes)):
        if time.time() < deadline - 5.0:
            routes[i] = two_opt_route(routes[i])

    # Verify all customers are covered
    all_served = set()
    for r in routes:
        for v in r:
            if v not in stations:
                all_served.add(v)

    if all_served != set(customers):
        # Fallback: one customer per route
        routes = [[v] for v in customers]

    # Final feasibility check
    for r in routes:
        _, feasible, _ = simulate_route(r)
        if not feasible:
            # Fallback
            routes = [[v] for v in customers]
            break

    return routes
# EVOLVE-BLOCK-END
