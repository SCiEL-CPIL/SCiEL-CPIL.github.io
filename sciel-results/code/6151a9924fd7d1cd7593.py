"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import math
import time

def plan(instance):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    coords = np.array(instance["coords"], dtype=np.float64)
    n = len(coords)
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    # Precompute distance matrix
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))

    customers = list(range(1, n_customers + 1))
    all_nodes = list(range(n))

    def route_cost(route):
        """Compute total distance of a route (depot at both ends)."""
        if not route:
            return 0.0
        total = dist[0, route[0]]
        for i in range(len(route) - 1):
            total += dist[route[i], route[i + 1]]
        total += dist[route[-1], 0]
        return total

    def is_feasible(route):
        """Check if a route is feasible (battery, time windows, capacity, horizon)."""
        if not route:
            return True
        # Check capacity
        total_demand = sum(demands[v] for v in route if v not in stations and v != 0)
        if total_demand > capacity + 1e-9:
            return False

        t = 0.0
        batt = battery_max
        prev = 0
        for v in route:
            d = dist[prev, v]
            t += d / speed
            batt -= d
            if batt < -1e-9:
                return False
            if v not in stations and v != 0:
                # Customer: time window check
                t = max(t, ready[v])
                if t > due[v] + 1e-9:
                    return False
                t += service[v]
            elif v in stations:
                # Charging station: recharge
                energy = battery_max - batt
                if energy > 1e-9:
                    t += energy * recharge_rate
                batt = battery_max
            # v == 0 (depot) shouldn't appear in route
            prev = v

        # Return to depot
        d = dist[prev, 0]
        t += d / speed
        batt -= d
        if batt < -1e-9:
            return False
        if t > horizon + 1e-9:
            return False
        return True

    def build_initial_routes():
        """Build initial solution using nearest-neighbor with time windows."""
        remaining = set(customers)
        routes = []
        # Sort customers by due time for initial ordering
        cust_order = sorted(customers, key=lambda v: due[v])

        while remaining:
            # Start a new route from depot
            route = []
            curr = 0
            t = 0.0
            batt = battery_max
            load = 0.0

            while remaining:
                best_node = -1
                best_dist = float('inf')
                for v in remaining:
                    if v in stations or v == 0:
                        continue
                    if load + demands[v] > capacity + 1e-9:
                        continue
                    d = dist[curr, v]
                    new_t = t + d / speed
                    new_batt = batt - d
                    if new_batt < -1e-9:
                        continue
                    # Check if we can reach customer within time window
                    # We might need to go through a station, but for simplicity skip that in init
                    arr_t = max(new_t, ready[v])
                    if arr_t > due[v] + 1e-9:
                        continue
                    if d < best_dist:
                        best_dist = d
                        best_node = v

                if best_node == -1:
                    # Try inserting a station before going to next customer
                    # For now, just break if no customer reachable
                    break

                d = dist[curr, best_node]
                t += d / speed
                batt -= d
                t = max(t, ready[best_node])
                t += service[best_node]
                batt -= 0  # service doesn't use battery
                load += demands[best_node]
                route.append(best_node)
                remaining.discard(best_node)
                curr = best_node

            if route:
                # Check feasibility; if not, try to fix by splitting
                if is_feasible(route):
                    routes.append(route)
                else:
                    # Fallback: one customer per route
                    for v in route:
                        routes.append([v])
                        remaining.add(v)  # re-add, but we already removed them... let's handle differently

        # Ensure all customers are covered
        served = set()
        for r in routes:
            for v in r:
                served.add(v)
        for v in customers:
            if v not in served:
                routes.append([v])

        return routes

    def local_search(routes, time_limit):
        """Apply 2-opt and swap moves to improve total distance while maintaining feasibility."""
        total_dist = sum(route_cost(r) for r in routes)
        best_routes = [list(r) for r in routes]
        best_dist = total_dist
        last_improvement = time.time()

        while time.time() < time_limit:
            improved = False
            n_routes = len(best_routes)

            # Try 2-opt within each route
            for ri in range(n_routes):
                route = best_routes[ri]
                if len(route) < 2:
                    continue
                for i in range(len(route) - 1):
                    for j in range(i + 1, len(route)):
                        # Reverse segment i..j
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        if is_feasible(new_route):
                            new_total = total_dist - route_cost(route) + route_cost(new_route)
                            if new_total < best_dist - 1e-9:
                                best_routes[ri] = new_route
                                best_dist = new_total
                                improved = True
                                last_improvement = time.time()
                                if time.time() > time_limit:
                                    return best_routes, best_dist

            # Try swapping customers between routes
            for ri in range(n_routes):
                for rj in range(ri + 1, n_routes):
                    route_i = best_routes[ri]
                    route_j = best_routes[rj]
                    for ii in range(len(route_i)):
                        for jj in range(len(route_j)):
                            # Swap route_i[ii] and route_j[jj]
                            new_i = list(route_i)
                            new_j = list(route_j)
                            new_i[ii], new_j[jj] = new_j[jj], new_i[ii]
                            if is_feasible(new_i) and is_feasible(new_j):
                                new_total = (best_dist - route_cost(route_i) - route_cost(route_j)
                                             + route_cost(new_i) + route_cost(new_j))
                                if new_total < best_dist - 1e-9:
                                    best_routes[ri] = new_i
                                    best_routes[rj] = new_j
                                    best_dist = new_total
                                    improved = True
                                    last_improvement = time.time()
                                    if time.time() > time_limit:
                                        return best_routes, best_dist

            # Try moving a customer from one route to another
            for ri in range(n_routes):
                route_i = best_routes[ri]
                for ii in range(len(route_i)):
                    for rj in range(n_routes):
                        if ri == rj:
                            continue
                        route_j = best_routes[rj]
                        for jj in range(len(route_j) + 1):
                            # Move route_i[ii] to position jj in route_j
                            new_i = list(route_i)
                            new_j = list(route_j)
                            v = new_i.pop(ii)
                            new_j.insert(jj, v)
                            if not new_i:
                                # Empty route, skip
                                continue
                            if is_feasible(new_i) and is_feasible(new_j):
                                new_total = (best_dist - route_cost(route_i) - route_cost(route_j)
                                             + route_cost(new_i) + route_cost(new_j))
                                if new_total < best_dist - 1e-9:
                                    best_routes[ri] = new_i
                                    best_routes[rj] = new_j
                                    best_dist = new_total
                                    improved = True
                                    last_improvement = time.time()
                                    if time.time() > time_limit:
                                        return best_routes, best_dist

            if not improved:
                break

        return best_routes, best_dist

    # Build initial solution
    routes = build_initial_routes()

    # Local search
    routes, _ = local_search(routes, deadline - 2.0)

    return routes
# EVOLVE-BLOCK-END
