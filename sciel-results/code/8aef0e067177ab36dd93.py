# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random
from itertools import permutations


def _build_dist_matrix(coords):
    n = len(coords)
    xs = np.array([c[0] for c in coords])
    ys = np.array([c[1] for c in coords])
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    return dist


def _route_cost(route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon):
    """
    Fast feasibility check and cost computation for a route.
    Returns (feasible, total_distance).
    """
    if not route:
        return True, 0.0

    # Demand check
    total_demand = 0
    for v in route:
        if v <= n_customers:
            total_demand += demands[v]
    if total_demand > capacity:
        return False, 0.0

    battery = battery_max
    time_t = 0.0
    total_dist = 0.0
    prev = 0

    for v in route:
        d = dist[prev][v]
        battery -= d
        total_dist += d
        time_t += d / speed

        if battery < -1e-9:
            return False, 0.0

        if v <= n_customers:
            if time_t < ready[v]:
                time_t = ready[v]
            if time_t > due[v] + 1e-9:
                return False, 0.0
            time_t += service[v]
        else:
            # Charging station
            energy_needed = battery_max - battery
            time_t += energy_needed * recharge_rate
            battery = battery_max

        prev = v

    # Return to depot
    d = dist[prev][0]
    battery -= d
    total_dist += d
    time_t += d / speed

    if battery < -1e-9 or time_t > horizon + 1e-9:
        return False, 0.0

    return True, total_dist


def _construct_savings_routes(instance, dist):
    """
    Construct initial routes using the Clarke-Wright Savings heuristic.
    """
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    battery_max = instance["battery"]
    capacity = instance["capacity"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    # Start with each customer as its own route
    routes = [[i] for i in range(1, n_customers + 1)]
    route_of = {c: i for i, r in enumerate(routes) for c in r}
    route_demand = [demands[c] for c in range(1, n_customers + 1)]

    # Calculate savings
    savings = []
    for i in range(n_customers):
        for j in range(i + 1, n_customers):
            ci = i + 1
            cj = j + 1
            s = dist[0][ci] + dist[0][cj] - dist[ci][cj]
            savings.append((s, ci, cj))
    savings.sort(reverse=True)

    for s, ci, cj in savings:
        ri = route_of.get(ci, -1)
        rj = route_of.get(cj, -1)
        if ri == -1 or rj == -1 or ri == rj:
            continue

        if route_demand[ri] + route_demand[rj] > capacity:
            continue

        # Try merging in both orders
        best_new_route = None
        best_dist = float('inf')
        for order in [[routes[ri], routes[rj]], [routes[rj], routes[ri]]]:
            merged = order[0] + order[1]
            feasible, total_dist = _route_cost(merged, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
            if feasible and total_dist < best_dist:
                best_dist = total_dist
                best_new_route = merged

        if best_new_route is not None:
            routes[ri] = best_new_route
            routes[rj] = []
            route_demand[ri] += route_demand[rj]
            route_demand[rj] = 0
            for c in best_new_route:
                route_of[c] = ri

    # Remove empty routes
    routes = [r for r in routes if r]

    # Ensure all customers are served (safety net)
    all_served = set()
    for r in routes:
        all_served.update(r)
    for c in range(1, n_customers + 1):
        if c not in all_served:
            routes.append([c])

    return routes


def _improve_route(route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon, deadline):
    """
    Apply 2-opt and Or-opt moves to a single route.
    """
    if len(route) < 2:
        return route

    n = len(route)
    improved = True
    passes = 0
    max_passes = 10

    while improved and passes < max_passes and time.time() < deadline:
        improved = False
        passes += 1

        # 2-Opt
        for i in range(n - 1):
            if time.time() > deadline:
                return route
            for j in range(i + 1, n):
                if time.time() > deadline:
                    return route
                prev_i = route[i - 1] if i > 0 else 0
                next_j = route[j + 1] if j < n - 1 else 0

                d_old = dist[prev_i][route[i]] + dist[route[j]][next_j]
                d_new = dist[prev_i][route[j]] + dist[route[i]][next_j]

                if d_new < d_old - 1e-9:
                    new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                    feasible, _ = _route_cost(new_route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
                    if feasible:
                        route = new_route
                        n = len(route)
                        improved = True
                        break
            if improved:
                break

        # Or-Opt (relocate 1 node)
        if not improved and len(route) >= 3:
            for i in range(n):
                if time.time() > deadline:
                    return route
                node = route[i]
                rest = route[:i] + route[i + 1:]
                
                for pos in range(len(rest) + 1):
                    if time.time() > deadline:
                        return route
                    if pos == i or (pos > i and pos <= i + 1):
                        continue

                    new_route = rest[:pos] + [node] + rest[pos:]
                    
                    # Quick distance check
                    if pos == 0:
                        if len(rest) > 0:
                            d_old = dist[0][rest[0]] + dist[rest[-1]][0]
                            d_new = dist[0][node] + dist[node][rest[0]] + dist[rest[-1]][0]
                        else:
                            d_old = 0
                            d_new = dist[0][node] + dist[node][0]
                    elif pos == len(rest):
                        if len(rest) > 0:
                            d_old = dist[rest[-1]][0]
                            d_new = dist[rest[-1]][node] + dist[node][0]
                        else:
                            d_old = 0
                            d_new = dist[0][node] + dist[node][0]
                    else:
                        if len(rest) > 0:
                            d_old = dist[rest[pos - 1]][rest[pos]]
                            d_new = dist[rest[pos - 1]][node] + dist[node][rest[pos]]
                        else:
                            d_old = 0
                            d_new = dist[0][node] + dist[node][0]

                    if d_new < d_old - 1e-9:
                        feasible, _ = _route_cost(new_route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
                        if feasible:
                            route = new_route
                            n = len(route)
                            improved = True
                            break
                if improved:
                    break

    return route


def _lns_solve(instance, dist, initial_routes, deadline):
    """
    Large Neighborhood Search (LNS) to improve the set of routes.
    """
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    battery_max = instance["battery"]
    capacity = instance["capacity"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    # Copy initial routes
    routes = [list(r) for r in initial_routes]
    
    # Calculate initial cost
    current_cost = sum(_route_cost(r, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)[1] for r in routes)
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        # 1. Destroy: Remove a random subset of customers
        all_customers = [c for r in routes for c in r if c <= n_customers]
        if not all_customers:
            break
            
        # Remove 10-30% of customers, at least 1
        k = max(1, int(len(all_customers) * rng.uniform(0.1, 0.3)))
        removed = rng.sample(all_customers, k)
        removed_set = set(removed)
        
        # Remove from routes
        new_routes = []
        for r in routes:
            new_r = [c for c in r if c not in removed_set]
            if new_r:
                new_routes.append(new_r)
        
        # 2. Repair: Re-insert removed customers
        # Sort removed customers by some heuristic (e.g., distance to depot) to make insertion easier?
        # Or just random order.
        rng.shuffle(removed)
        
        feasible_new_routes = True
        for c in removed:
            best_pos = -1
            best_cost = float('inf')
            best_route_idx = -1
            
            # Try inserting into existing routes
            for idx, r in enumerate(new_routes):
                current_demand = sum(demands[v] for v in r if v <= n_customers)
                if current_demand + demands[c] > capacity:
                    continue
                
                for pos in range(len(r) + 1):
                    new_r = r[:pos] + [c] + r[pos:]
                    feasible, total_dist = _route_cost(new_r, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
                    if feasible:
                        # Calculate incremental cost
                        if pos == 0:
                            if len(r) > 0:
                                inc = dist[0][c] + dist[c][r[0]] - dist[0][r[0]]
                            else:
                                inc = dist[0][c] + dist[c][0]
                        elif pos == len(r):
                            inc = dist[r[-1]][c] + dist[c][0] - dist[r[-1]][0]
                        else:
                            inc = dist[r[pos - 1]][c] + dist[c][r[pos]] - dist[r[pos - 1]][r[pos]]
                        
                        if inc < best_cost:
                            best_cost = inc
                            best_pos = pos
                            best_route_idx = idx
            
            # Try starting a new route
            new_route = [c]
            feasible, total_dist = _route_cost(new_route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
            if feasible:
                inc = total_dist
                if inc < best_cost:
                    best_cost = inc
                    best_pos = 0
                    best_route_idx = -1 # Indicates new route
            
            if best_route_idx == -1:
                # New route
                new_routes.append([c])
            else:
                # Insert into existing
                r = new_routes[best_route_idx]
                new_routes[best_route_idx] = r[:best_pos] + [c] + r[best_pos:]
            
            if not feasible:
                feasible_new_routes = False
                break
        
        if not feasible_new_routes:
            continue
            
        # 3. Evaluate new solution
        new_cost = sum(_route_cost(r, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)[1] for r in new_routes)
        
        # Accept if better (Hill Climbing)
        if new_cost < current_cost - 1e-9:
            routes = new_routes
            current_cost = new_cost
            
            # Local improvement on the new routes
            # Run a quick 2-opt on each route
            for i in range(len(routes)):
                routes[i] = _improve_route(routes[i], dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon, time.time() + 0.5)
                # Recalculate cost after improvement
            current_cost = sum(_route_cost(r, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)[1] for r in routes)
            
    return routes


def _solve_instance(instance, time_limit=45):
    """
    Solve a single instance using LNS.
    """
    start_time = time.time()
    deadline = start_time + time_limit

    coords = instance["coords"]
    dist = _build_dist_matrix(coords)

    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    battery_max = instance["battery"]
    capacity = instance["capacity"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    # Phase 1: Initial Solution via Savings
    routes = _construct_savings_routes(instance, dist)
    
    # Phase 2: Initial Local Search
    for i in range(len(routes)):
        if time.time() > deadline - 5:
            break
        routes[i] = _improve_route(routes[i], dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon, deadline - 5)

    # Phase 3: LNS
    if time.time() < deadline - 2:
        routes = _lns_solve(instance, dist, routes, deadline - 2)

    # Final cleanup and verification
    valid_routes = []
    all_served = set()
    for r in routes:
        if not r:
            continue
        feasible, _ = _route_cost(r, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
        if feasible:
            valid_routes.append(r)
            all_served.update(r)

    # Ensure all customers are served
    for c in range(1, n_customers + 1):
        if c not in all_served:
            valid_routes.append([c])

    return valid_routes


def plan(instance):
    """Main entry point. Returns routes for the instance."""
    routes = _solve_instance(instance, time_limit=45)
    return routes
# EVOLVE-BLOCK-END
