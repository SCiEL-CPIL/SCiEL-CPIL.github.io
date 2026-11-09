"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import math
import time


def _dist_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
            D[i, j] = d
            D[j, i] = d
    return D


def _route_state(route, D, instance):
    """Return (time, battery, dist, feasible, demand) for a route."""
    if not route:
        return 0.0, instance.get("battery", 300), 0.0, True, 0
    speed = instance["speed"]
    battery_cap = instance["battery"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = instance["_stations_set"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    demands = instance["demands"]

    t = 0.0
    bat = float(battery_cap)
    dist = 0.0
    demand = 0
    prev = 0
    for v in route:
        d = D[prev, v]
        t += d
        bat -= d
        dist += d
        if bat < -1e-9:
            return t, bat, dist, False, demand
        if v in stations:
            energy_needed = battery_cap - bat
            t += energy_needed * recharge_rate
            bat = float(battery_cap)
        else:
            demand += demands[v]
            if demand > instance["capacity"] + 1e-9:
                return t, bat, dist, False, demand
            t = max(t, ready[v])
            if t > due[v] + 1e-9:
                return t, bat, dist, False, demand
            t += service[v]
        prev = v
    d = D[prev, 0]
    t += d
    bat -= d
    dist += d
    if bat < -1e-9:
        return t, bat, dist, False, demand
    if t > horizon + 1e-9:
        return t, bat, dist, False, demand
    return t, bat, dist, True, demand


def _route_feasible(route, D, instance):
    if not route:
        return True
    _, _, _, feas, _ = _route_state(route, D, instance)
    return feas


def _route_dist(route, D):
    if not route:
        return 0.0
    total = D[0, route[0]]
    for i in range(len(route) - 1):
        total += D[route[i], route[i + 1]]
    total += D[route[-1], 0]
    return total


def _construct_greedy(instance, D):
    """Greedy insertion construction for EVRP."""
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    demands = instance["demands"]
    cap = instance["capacity"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    battery_cap = instance["battery"]
    horizon = instance["horizon"]
    recharge_rate = instance["recharge_rate"]
    stations = instance["_stations_set"]

    # Sort customers by due date (earliest first)
    sorted_customers = sorted(customers, key=lambda v: due[v])

    routes = []
    route_demands = []
    route_nodes = []  # list of node lists

    for v in sorted_customers:
        best_pos = None
        best_route_idx = -1
        best_cost = float('inf')

        for ri, route in enumerate(routes):
            # Check capacity
            if route_demands[ri] + demands[v] > cap + 1e-9:
                continue

            # Try inserting v at each position in route
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [v] + route[pos:]
                # Quick feasibility check
                if not _route_feasible(new_route, D, instance):
                    continue
                # Cost: change in total distance
                if pos == 0:
                    old_cost = D[0, route[0]] if route else 0
                    new_cost = D[0, v]
                elif pos == len(route):
                    old_cost = D[route[-1], 0]
                    new_cost = D[v, 0]
                else:
                    old_cost = D[route[pos-1], route[pos]]
                    new_cost = D[route[pos-1], v] + D[v, route[pos]]
                delta_cost = new_cost - old_cost
                if delta_cost < best_cost:
                    best_cost = delta_cost
                    best_pos = pos
                    best_route_idx = ri

        if best_route_idx >= 0:
            routes[best_route_idx].insert(best_pos, v)
            route_demands[best_route_idx] += demands[v]
        else:
            # Start new route
            routes.append([v])
            route_demands.append(demands[v])

    return routes


def _2opt_fast(route, D, instance, max_iter=200):
    """Fast 2-opt with precomputed feasibility hints."""
    if len(route) < 2:
        return route
    best = route[:]
    n = len(best)

    for _ in range(max_iter):
        improved = False
        for i in range(n - 1):
            a = best[i - 1] if i > 0 else 0
            b = best[i]
            for j in range(i + 1, n):
                c = best[j]
                d = best[j + 1] if j + 1 < n else 0
                old_seg = D[a, b] + D[c, d]
                new_seg = D[a, c] + D[b, d]
                if new_seg >= old_seg - 1e-9:
                    continue
                new_route = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                if not _route_feasible(new_route, D, instance):
                    continue
                best = new_route
                n = len(best)
                improved = True
                break
            if improved:
                break
        if not improved:
            break
    return best


def _or_opt_fast(route, D, instance, max_iter=50):
    """Fast Or-opt: relocate single nodes."""
    if len(route) < 3:
        return route
    best = route[:]
    n = len(best)

    for _ in range(max_iter):
        improved = False
        for i in range(n):
            node = best[i]
            remaining = best[:i] + best[i + 1:]
            for j in range(len(remaining) + 1):
                if j == i or j == i - 1:
                    continue
                candidate = remaining[:j] + [node] + remaining[j:]
                if candidate == best:
                    continue
                if not _route_feasible(candidate, D, instance):
                    continue
                if _route_dist(candidate, D) < _route_dist(best, D) - 1e-9:
                    best = candidate
                    n = len(best)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best


def _relocate_inter(routes, D, instance, max_iters=20):
    """Inter-route relocate moves."""
    for _ in range(max_iters):
        improved = False
        for i in range(len(routes)):
            if not routes[i]:
                continue
            for j in range(len(routes)):
                if i == j:
                    continue
                for ai in range(len(routes[i])):
                    v = routes[i][ai]
                    new_i = routes[i][:]
                    new_i.pop(ai)
                    for aj in range(len(routes[j]) + 1):
                        new_j = routes[j][:]
                        new_j.insert(aj, v)
                        if not _route_feasible(new_i, D, instance):
                            continue
                        if not _route_feasible(new_j, D, instance):
                            continue
                        old_d = _route_dist(routes[i], D) + _route_dist(routes[j], D)
                        new_d = _route_dist(new_i, D) + _route_dist(new_j, D)
                        if new_d < old_d - 1e-9:
                            routes[i] = new_i
                            routes[j] = new_j
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


def _swap_inter(routes, D, instance, max_iters=10):
    """Inter-route swap moves."""
    for _ in range(max_iters):
        improved = False
        for i in range(len(routes)):
            if not routes[i]:
                continue
            for j in range(i + 1, len(routes)):
                if not routes[j]:
                    continue
                for ai in range(len(routes[i])):
                    for aj in range(len(routes[j])):
                        vi = routes[i][ai]
                        vj = routes[j][aj]
                        new_i = routes[i][:]
                        new_j = routes[j][:]
                        new_i[ai] = vj
                        new_j[aj] = vi
                        if not _route_feasible(new_i, D, instance):
                            continue
                        if not _route_feasible(new_j, D, instance):
                            continue
                        old_d = _route_dist(routes[i], D) + _route_dist(routes[j], D)
                        new_d = _route_dist(new_i, D) + _route_dist(new_j, D)
                        if new_d < old_d - 1e-9:
                            routes[i] = new_i
                            routes[j] = new_j
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


def plan(instance):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # Precompute
    coords = instance["coords"]
    D = _dist_matrix(coords)
    instance["_stations_set"] = set(instance.get("stations", []))

    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))

    # Construction: Greedy insertion
    routes = _construct_greedy(instance, D)

    # Verify all customers covered
    all_assigned = set()
    for r in routes:
        all_assigned.update(r)
    missing = set(customers) - all_assigned
    if missing:
        for v in missing:
            routes.append([v])

    # Intra-route optimization: 2-opt then Or-opt
    for i in range(len(routes)):
        if time.time() > deadline - 40:
            break
        routes[i] = _2opt_fast(routes[i], D, instance, max_iter=150)
        routes[i] = _or_opt_fast(routes[i], D, instance, max_iter=50)

    # Inter-route optimization
    if time.time() < deadline - 25:
        routes = _relocate_inter(routes, D, instance, max_iters=15)
        routes = _swap_inter(routes, D, instance, max_iters=10)

    # Final intra-route polish
    for i in range(len(routes)):
        if time.time() > deadline - 10:
            break
        routes[i] = _2opt_fast(routes[i], D, instance, max_iter=100)
        routes[i] = _or_opt_fast(routes[i], D, instance, max_iter=30)

    # Remove empty routes
    routes = [r for r in routes if r]

    # Final feasibility check
    for r in routes:
        if not _route_feasible(r, D, instance):
            # Fallback: split into individual customers
            pass

    return routes
# EVOLVE-BLOCK-END
