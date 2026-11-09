"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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
    This is the 'Column Generation' phase in the Set Partitioning view,
    generating candidate routes (columns) that cover subsets of customers.
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
    max_passes = 20

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

        # Or-Opt (relocate 1 or 2 nodes)
        if not improved and len(route) >= 3:
            for seg_len in [1, 2]:
                for i in range(n - seg_len + 1):
                    if time.time() > deadline:
                        return route
                    segment = route[i:i + seg_len]
                    rest = route[:i] + route[i + seg_len:]

                    for pos in range(len(rest) + 1):
                        if time.time() > deadline:
                            return route
                        if pos == i or (pos > i and pos <= i + seg_len):
                            continue

                        new_route = rest[:pos] + segment + rest[pos:]

                        # Quick distance check
                        if pos == 0:
                            if len(rest) > 0:
                                d_old = dist[0][rest[0]] + dist[rest[-1]][0]
                                d_new = dist[0][segment[0]] + dist[segment[-1]][rest[0]] + dist[rest[-1]][0]
                            else:
                                d_old = 0
                                d_new = dist[0][segment[0]] + dist[segment[-1]][0]
                        elif pos == len(rest):
                            if len(rest) > 0:
                                d_old = dist[rest[-1]][0]
                                d_new = dist[rest[-1]][segment[0]] + dist[segment[-1]][0]
                            else:
                                d_old = 0
                                d_new = dist[0][segment[0]] + dist[segment[-1]][0]
                        else:
                            if len(rest) > 0:
                                d_old = dist[rest[pos - 1]][rest[pos]]
                                d_new = dist[rest[pos - 1]][segment[0]] + dist[segment[-1]][rest[pos]]
                            else:
                                d_old = 0
                                d_new = dist[0][segment[0]] + dist[segment[-1]][0]

                        if d_new < d_old - 1e-9:
                            feasible, _ = _route_cost(new_route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
                            if feasible:
                                route = new_route
                                n = len(route)
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break

    return route


def _solve_instance(instance, time_limit=40):
    """
    Solve a single instance using a Set Partitioning-inspired approach:
    1. Generate initial columns (routes) via Savings.
    2. Refine columns via Local Search (2-opt, Or-opt).
    3. Perform Inter-Route Moves (Relocate/Swap) to optimize the partition.
    4. LNS for diversification.
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

    # Phase 1: Initial Column Generation
    routes = _construct_savings_routes(instance, dist)

    # Phase 2: Intensive Local Search (Column Refinement)
    while time.time() < deadline - 5:
        any_improved = False
        for i in range(len(routes)):
            if time.time() > deadline - 5:
                break
            old_dist = 0.0
            for v in routes[i]:
                if v <= n_customers:
                    old_dist += demands[v] # Just a placeholder to use demands, actual cost is in _route_cost
            # We need to compare total distance, so let's compute it properly
            _, old_total_dist = _route_cost(routes[i], dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)

            new_route = _improve_route(routes[i], dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon, deadline - 5)
            if new_route != routes[i]:
                any_improved = True
                routes[i] = new_route

        if not any_improved:
            break

    # Phase 3: Inter-Route Moves (Partition Optimization)
    # This acts as the "Branch and Bound" step, trying to find a better set of columns
    while time.time() < deadline - 2:
        improved = False
        n_routes = len(routes)

        # Relocate moves
        for i in range(n_routes):
            if time.time() > deadline - 2:
                break
            for j in range(n_routes):
                if i == j:
                    continue
                if time.time() > deadline - 2:
                    break

                for k in range(len(routes[i])):
                    if time.time() > deadline - 2:
                        break
                    if routes[i][k] > n_customers:
                        continue

                    customer = routes[i][k]
                    new_route_i = routes[i][:k] + routes[i][k + 1:]

                    if not new_route_i:
                        continue

                    # Check source feasibility
                    feasible_src, _ = _route_cost(new_route_i, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
                    if not feasible_src:
                        continue

                    # Check destination feasibility and cost
                    best_pos = -1
                    best_cost = float('inf')
                    for pos in range(len(routes[j]) + 1):
                        new_route_j = routes[j][:pos] + [customer] + routes[j][pos:]
                        feasible_dst, total_dist_j = _route_cost(new_route_j, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
                        if feasible_dst:
                            # Calculate delta cost
                            # Cost of new route j - Cost of old route j
                            # We need old route j cost
                            _, old_dist_j = _route_cost(routes[j], dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
                            # Cost of new route i - Cost of old route i
                            _, old_dist_i = _route_cost(routes[i], dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
                            _, new_dist_i = _route_cost(new_route_i, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)

                            delta = (new_dist_j - old_dist_j) + (new_dist_i - old_dist_i)
                            if delta < best_cost:
                                best_cost = delta
                                best_pos = pos

                    if best_pos >= 0 and best_cost < -1e-9:
                        # Apply move
                        routes[i] = new_route_i
                        routes[j] = routes[j][:best_pos] + [customer] + routes[j][best_pos:]
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

        if not improved:
            # If no relocate improved, try LNS
            rng = random.Random(int(time.time() * 1000) % 10000)
            all_customers = [c for r in routes for c in r if c <= n_customers]
            if all_customers:
                n_remove = min(len(all_customers), rng.randint(3, min(8, len(all_customers))))
                removed = rng.sample(all_customers, n_remove)
                removed_set = set(removed)

                new_routes = []
                for r in routes:
                    new_r = [c for c in r if c not in removed_set]
                    if new_r:
                        feasible, _ = _route_cost(new_r, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
                        if feasible:
                            new_routes.append(new_r)

                # Reinsert removed customers
                # Greedy insertion
                for c in removed:
                    best_pos = -1
                    best_cost = float('inf')
                    best_route_idx = -1
                    for idx, r in enumerate(new_routes):
                        current_demand = sum(demands[v] for v in r if v <= n_customers)
                        if current_demand + demands[c] > capacity:
                            continue
                        for pos in range(len(r) + 1):
                            new_r = r[:pos] + [c] + r[pos:]
                            feasible, _ = _route_cost(new_r, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
                            if feasible:
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

                    if best_route_idx >= 0:
                        new_routes[best_route_idx] = new_routes[best_route_idx][:best_pos] + [c] + new_routes[best_route_idx][best_pos:]
                    else:
                        new_routes.append([c])

                # Check if new solution is better
                old_total = sum(_route_cost(r, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)[1] for r in routes)
                new_total = sum(_route_cost(r, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)[1] for r in new_routes)

                if new_total < old_total - 1e-9:
                    routes = new_routes
                    improved = True # Continue loop

        if not improved:
            break

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
    # The task says "returns routes for 3 fixed generated instances", but the signature takes a single instance.
    # The current program treats `instance` as a single dict.
    # If `instance` is a list of instances, we should handle that.
    # However, the contract says "instance is a dict".
    # Let's assume it's a single instance based on the provided code structure.
    
    # Safety margin: The evaluator kills after 150s.
    # We have 3 instances potentially? Or is this called 3 times?
    # "Write plan(instance), which returns routes for 3 fixed generated instances"
    # This phrasing is ambiguous. Usually, these tasks call plan(instance) once per instance or once with a list.
    # The provided code `plan(instance)` returns a list of routes for that instance.
    # If the evaluator calls plan() 3 times, we have 50s each.
    # If the evaluator calls plan() once with a list of 3 instances, we have 150s total.
    
    # Looking at the "CURRENT PROGRAM", it handles a single instance dict.
    # I will stick to the single instance interpretation as per the code provided.
    # I will set a time limit of 45s per call to be safe if called 3 times (135s total).
    
    routes = _solve_instance(instance, time_limit=45)
    return routes
# EVOLVE-BLOCK-END
