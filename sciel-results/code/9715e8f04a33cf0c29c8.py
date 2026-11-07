# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations


def _euclid(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return (dx * dx + dy * dy) ** 0.5


def _check_route(route, instance):
    """Check if a route (list of node ids, depot implicit at ends) is feasible.
    Returns (feasible, total_distance, total_time)."""
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])

    # Total demand check
    total_demand = sum(demands[n] for n in route if demands[n] > 0)
    if total_demand > capacity:
        return False, 0, 0

    # Simulate
    current_node = 0
    current_time = 0.0
    current_battery = float(battery)
    total_distance = 0.0

    for n in route:
        d = _euclid(coords[current_node], coords[n])
        total_distance += d
        current_time += d / speed
        current_battery -= d

        if current_battery < -1e-9:
            return False, 0, 0

        if n in stations:
            # Recharge
            energy_needed = battery - current_battery
            if energy_needed > 0:
                current_time += energy_needed * recharge_rate
            current_battery = float(battery)
        else:
            # Customer service
            if current_time < ready[n]:
                current_time = float(ready[n])
            if current_time > due[n] + 1e-9:
                return False, 0, 0
            current_time += service[n]

        current_node = n

    # Return to depot
    d = _euclid(coords[current_node], coords[0])
    total_distance += d
    current_time += d / speed
    current_battery -= d

    if current_battery < -1e-9:
        return False, 0, 0
    if current_time > horizon + 1e-9:
        return False, 0, 0

    return True, total_distance, current_time


def _route_distance(route, instance):
    """Compute total Euclidean distance of a route including depot legs."""
    coords = instance["coords"]
    if not route:
        return 0.0
    total = _euclid(coords[0], coords[route[0]])
    for i in range(len(route) - 1):
        total += _euclid(coords[route[i]], coords[route[i + 1]])
    total += _euclid(coords[route[-1]], coords[0])
    return total


def plan(instance):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    coords = np.array(instance["coords"], dtype=float)
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = instance["stations"]
    station_set = set(stations)
    n_nodes = len(coords)

    # Precompute distance matrix
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            dist[i, j] = _euclid(coords[i], coords[j])

    # Sort customers by due time for initial construction
    sorted_customers = sorted(customers, key=lambda c: due[c])

    # --- Phase 1: Greedy construction ---
    # Try to insert customers into existing routes, starting with empty routes
    # We'll use a simple approach: start with one route per customer, then merge/optimize

    # Initial: each customer in its own route
    routes = [[c] for c in sorted_customers]

    # Merge routes: try to combine pairs that are close together
    def _can_merge(r1, r2, instance):
        """Check if we can merge route r1 and r2 (concatenation) feasibly."""
        # Try r1 + r2 and r2 + r1
        for combined in [r1 + r2, r2 + r1]:
            feasible, _, _ = _check_route(combined, instance)
            if feasible:
                return True
        return False

    # Repeatedly try to merge closest pairs
    merged = True
    while merged and time.time() < deadline:
        merged = False
        best_pair = None
        best_dist = float('inf')
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                # Compute min distance between any node in route i and route j
                min_d = float('inf')
                for a in routes[i]:
                    for b in routes[j]:
                        d = dist[a, b]
                        if d < min_d:
                            min_d = d
                if min_d < best_dist:
                    best_dist = min_d
                    best_pair = (i, j)

        if best_pair is None:
            break

        i, j = best_pair
        if _can_merge(routes[i], routes[j], instance):
            # Try both orders, pick the one with less total distance
            combo1 = routes[i] + routes[j]
            combo2 = routes[j] + routes[i]
            d1 = _route_distance(combo1, instance)
            d2 = _route_distance(combo2, instance)
            if d1 <= d2:
                new_route = combo1
            else:
                new_route = combo2
            # Remove the two routes and add the merged one
            routes.pop(j)
            routes.pop(i)
            routes.append(new_route)
            merged = True

    # --- Phase 2: Local search (2-opt + relocate) ---
    def _improve_routes(routes, instance, max_iters=50):
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1

            # 2-opt within each route
            for r_idx in range(len(routes)):
                route = routes[r_idx]
                if len(route) < 2:
                    continue
                route_improved = True
                while route_improved and time.time() < deadline:
                    route_improved = False
                    for i in range(len(route) - 1):
                        for j in range(i + 1, len(route)):
                            # Reverse segment i..j
                            new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                            feasible, _, _ = _check_route(new_route, instance)
                            if feasible:
                                old_d = _route_distance(route, instance)
                                new_d = _route_distance(new_route, instance)
                                if new_d < old_d - 1e-9:
                                    routes[r_idx] = new_route
                                    route_improved = True
                                    improved = True
                                    break
                        if route_improved:
                            break

            # Relocate: move a customer from one route to another
            for r1 in range(len(routes)):
                for r2 in range(len(routes)):
                    if r1 == r2:
                        continue
                    route1 = routes[r1]
                    route2 = routes[r2]
                    for pos in range(len(route1)):
                        cust = route1[pos]
                        # Remove from route1
                        new_route1 = route1[:pos] + route1[pos+1:]
                        # Try inserting into route2 at each position
                        for ins_pos in range(len(route2) + 1):
                            new_route2 = route2[:ins_pos] + [cust] + route2[ins_pos:]
                            f1, _, _ = _check_route(new_route1, instance)
                            f2, _, _ = _check_route(new_route2, instance)
                            if f1 and f2:
                                old_d = _route_distance(route1, instance) + _route_distance(route2, instance)
                                new_d = _route_distance(new_route1, instance) + _route_distance(new_route2, instance)
                                if new_d < old_d - 1e-9:
                                    routes[r1] = new_route1
                                    routes[r2] = new_route2
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

            # Swap: exchange two customers between routes
            for r1 in range(len(routes)):
                for r2 in range(len(routes)):
                    if r1 == r2:
                        continue
                    route1 = routes[r1]
                    route2 = routes[r2]
                    for p1 in range(len(route1)):
                        for p2 in range(len(route2)):
                            c1 = route1[p1]
                            c2 = route2[p2]
                            new_route1 = route1[:p1] + [c2] + route1[p1+1:]
                            new_route2 = route2[:p2] + [c1] + route2[p2+1:]
                            f1, _, _ = _check_route(new_route1, instance)
                            f2, _, _ = _check_route(new_route2, instance)
                            if f1 and f2:
                                old_d = _route_distance(route1, instance) + _route_distance(route2, instance)
                                new_d = _route_distance(new_route1, instance) + _route_distance(new_route2, instance)
                                if new_d < old_d - 1e-9:
                                    routes[r1] = new_route1
                                    routes[r2] = new_route2
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

        return routes

    routes = _improve_routes(routes, instance, max_iters=30)

    # --- Phase 3: Try adding charging stations to routes if needed ---
    # Check if any route is infeasible; if so, try inserting stations
    def _fix_route(route, instance):
        """Try to make a route feasible by inserting charging stations."""
        if not route:
            return route
        feasible, _, _ = _check_route(route, instance)
        if feasible:
            return route

        # Try inserting a station at various positions
        best_new_route = None
        best_d = float('inf')
        for st in stations:
            for pos in range(len(route) + 1):
                candidate = route[:pos] + [st] + route[pos:]
                f, d, _ = _check_route(candidate, instance)
                if f and d < best_d:
                    best_d = d
                    best_new_route = candidate

        if best_new_route is not None:
            return best_new_route

        # Try two stations
        for st1 in stations:
            for st2 in stations:
                if st1 == st2:
                    continue
                for pos1 in range(len(route) + 1):
                    temp = route[:pos1] + [st1] + route[pos1:]
                    for pos2 in range(pos1 + 1, len(temp) + 1):
                        candidate = temp[:pos2] + [st2] + temp[pos2:]
                        f, d, _ = _check_route(candidate, instance)
                        if f and d < best_d:
                            best_d = d
                            best_new_route = candidate
        return best_new_route if best_new_route is not None else route

    fixed_routes = []
    for r in routes:
        fixed = _fix_route(r, instance)
        if fixed:
            fixed_routes.append(fixed)

    # Verify all customers are covered exactly once
    all_served = []
    for r in fixed_routes:
        for n in r:
            if n not in station_set:
                all_served.append(n)

    # If some customers are missing, add them as single-customer routes
    missing = set(customers) - set(all_served)
    for c in missing:
        fixed_routes.append([c])

    # Final improvement pass
    if time.time() < deadline - 5.0:
        fixed_routes = _improve_routes(fixed_routes, instance, max_iters=10)

    # Final feasibility check
    final_routes = []
    for r in fixed_routes:
        if not r:
            continue
        feasible, _, _ = _check_route(r, instance)
        if feasible:
            final_routes.append(r)
        else:
            # Fallback: split into individual customers
            for c in r:
                if c not in station_set:
                    final_routes.append([c])

    # Ensure all customers are covered
    served = set()
    for r in final_routes:
        for n in r:
            if n not in station_set:
                served.add(n)
    for c in customers:
        if c not in served:
            final_routes.append([c])

    return final_routes
# EVOLVE-BLOCK-END
