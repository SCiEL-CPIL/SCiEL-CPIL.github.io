"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time
from itertools import combinations

_DEADLINE = 115.0  # wall-clock safety margin


def _dist(p, q):
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return float(np.hypot(dx, dy))


def _build_dist_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = _dist(coords[i], coords[j])
            D[j, i] = D[i, j]
    return D


def _simulate_route(route, D, demands, ready, due, service, n_customers,
                    stations, battery_max, recharge_rate, horizon, speed=1.0):
    """Simulate a route and return (feasible, total_distance).
    route is a list of customer node ids (1..n_customers).
    Depot is node 0. Stations may be inserted by the caller if needed.
    Here we check feasibility assuming NO station stops; if infeasible,
    we return (False, inf). The caller can add station stops."""
    if not route:
        return True, 0.0
    total_d = 0.0
    time_cur = 0.0
    batt = battery_max
    # start at depot
    prev = 0
    for node in route:
        d = D[prev][node]
        batt -= d
        if batt < -1e-9:
            return False, float('inf')
        time_cur += d / speed
        total_d += d
        if node <= n_customers:
            time_cur = max(time_cur, ready[node])
            if time_cur > due[node] + 1e-9:
                return False, float('inf')
            time_cur += service[node]
        else:
            # station
            recharge_time = (battery_max - batt) * recharge_rate
            time_cur += recharge_time
            batt = battery_max
        prev = node
    # return to depot
    d = D[prev][0]
    batt -= d
    if batt < -1e-9:
        return False, float('inf')
    time_cur += d / speed
    total_d += d
    if time_cur > horizon + 1e-9:
        return False, float('inf')
    return True, total_d


def _insert_station(route, D, demands, ready, due, service, n_customers,
                    stations, battery_max, recharge_rate, horizon, speed=1.0):
    """Try to make route feasible by inserting stations. Return (feasible, total_distance, new_route_with_stations) or (False, inf, None)."""
    # Simple approach: try inserting the nearest station after each leg where battery would go negative
    # We simulate and track where battery goes negative, then insert station there.
    # For simplicity, try a few strategies:
    # 1. No stations
    # 2. Station after every k customers
    # 3. Greedy: insert station before battery goes negative

    # First try without stations
    feasible, total_d = _simulate_route(route, D, demands, ready, due, service,
                                        n_customers, stations, battery_max,
                                        recharge_rate, horizon, speed)
    if feasible:
        return True, total_d, list(route)

    # Try inserting stations greedily
    # We'll build a new route with stations
    for attempt in range(5):
        new_route = []
        prev = 0
        batt = battery_max
        time_cur = 0.0
        total_d = 0.0
        feasible = True
        for node in route:
            d = D[prev][node]
            if batt - d < -1e-9:
                # Need to recharge; find best station
                best_st = None
                best_cost = float('inf')
                for st in stations:
                    d_to_st = D[prev][st]
                    if batt - d_to_st < -1e-9:
                        continue
                    recharge_time = (battery_max - (batt - d_to_st)) * recharge_rate
                    d_st_to_node = D[st][node]
                    cost = d_to_st + d_st_to_node
                    if cost < best_cost:
                        best_cost = cost
                        best_st = st
                if best_st is None:
                    feasible = False
                    break
                # Go to station
                d_to_st = D[prev][best_st]
                batt -= d_to_st
                time_cur += d_to_st / speed
                total_d += d_to_st
                recharge_time = (battery_max - batt) * recharge_rate
                time_cur += recharge_time
                batt = battery_max
                new_route.append(best_st)
                prev = best_st
                # Then go to node
                d = D[prev][node]
                batt -= d
                if batt < -1e-9:
                    feasible = False
                    break
                time_cur += d / speed
                total_d += d
                prev = node
            else:
                batt -= d
                time_cur += d / speed
                total_d += d
                prev = node

            if node <= n_customers:
                time_cur = max(time_cur, ready[node])
                if time_cur > due[node] + 1e-9:
                    feasible = False
                    break
                time_cur += service[node]

        if not feasible:
            continue

        # Return to depot
        d = D[prev][0]
        batt -= d
        if batt < -1e-9:
            feasible = False
            continue
        time_cur += d / speed
        total_d += d
        if time_cur > horizon + 1e-9:
            feasible = False
            continue

        if feasible:
            return True, total_d, new_route
    return False, float('inf'), None


def plan(instance):
    t_start = _time.time()
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    speed = 1.0

    D = _build_dist_matrix(coords)
    customers = list(range(1, n_customers + 1))

    # Greedy construction: nearest-neighbor with feasibility check
    # Sort customers by due time as a tiebreaker for ties in distance
    unvisited = set(customers)
    routes = []

    while unvisited:
        if _time.time() - t_start > _DEADLINE - 5:
            break
        # Start a new route from depot
        best_customer = None
        best_dist = float('inf')
        for c in unvisited:
            d = D[0][c]
            if d < best_dist:
                best_dist = d
                best_customer = c
        if best_customer is None:
            break

        route = [best_customer]
        unvisited.remove(best_customer)
        demand = demands[best_customer]
        prev = best_customer

        while unvisited:
            if _time.time() - t_start > _DEADLINE - 5:
                break
            # Find next customer to insert
            best_next = None
            best_nd = float('inf')
            for c in unvisited:
                if demand + demands[c] > capacity:
                    continue
                d = D[prev][c]
                if d < best_nd:
                    best_nd = d
                    best_next = c
            if best_next is None:
                break

            # Check feasibility of adding best_next
            trial_route = route + [best_next]
            feasible, total_d, _ = _insert_station(trial_route, D, demands, ready,
                                                    due, service, n_customers,
                                                    stations, battery_max,
                                                    recharge_rate, horizon, speed)
            if not feasible:
                break

            route = trial_route
            unvisited.remove(best_next)
            demand += demands[best_next]
            prev = best_next

        routes.append(route)

    # If we ran out of time, fallback: one customer per route for remaining
    if unvisited and _time.time() - t_start > _DEADLINE - 5:
        for c in sorted(unvisited, key=lambda x: due[x]):
            routes.append([c])
        unvisited = set()

    # Local search: 2-opt and relocation within each route, and inter-route moves
    # Keep it simple: try to improve by moving customers between adjacent positions
    improved = True
    max_iters = 50
    it = 0
    while improved and it < max_iters and _time.time() - t_start < _DEADLINE - 10:
        improved = False
        it += 1

        # Compute current total distance
        route_dists = []
        for r in routes:
            if not r:
                route_dists.append(0.0)
                continue
            # Find best feasible arrangement (we just simulate)
            feasible, total_d, _ = _insert_station(r, D, demands, ready, due,
                                                    service, n_customers, stations,
                                                    battery_max, recharge_rate,
                                                    horizon, speed)
            route_dists.append(total_d if feasible else float('inf'))
        current_total = sum(route_dists)

        # Try 2-opt within each route
        for i, r in enumerate(routes):
            if len(r) < 2:
                continue
            n = len(r)
            for a in range(n - 1):
                if _time.time() - t_start > _DEADLINE - 10:
                    break
                for b in range(a + 1, n):
                    new_r = r[:a + 1] + r[a + 1:b + 1][::-1] + r[b + 1:]
                    feasible, new_d, _ = _insert_station(new_r, D, demands, ready,
                                                          due, service, n_customers,
                                                          stations, battery_max,
                                                          recharge_rate, horizon, speed)
                    if feasible and new_d < route_dists[i] - 1e-6:
                        routes[i] = new_r
                        route_dists[i] = new_d
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

        if improved:
            continue

        # Try moving a customer from one route to another
        for i in range(len(routes)):
            if not routes[i]:
                continue
            for j in range(len(routes)):
                if i == j:
                    continue
                if _time.time() - t_start > _DEADLINE - 10:
                    break
                for pos_i, c in enumerate(routes[i]):
                    # Remove c from route i
                    r_i_without = routes[i][:pos_i] + routes[i][pos_i + 1:]
                    feasible_i, d_i, _ = _insert_station(r_i_without, D, demands,
                                                          ready, due, service,
                                                          n_customers, stations,
                                                          battery_max, recharge_rate,
                                                          horizon, speed)
                    if not feasible_i:
                        continue
                    # Try inserting c into route j at various positions
                    best_j_d = float('inf')
                    best_pos = -1
                    r_j = routes[j]
                    for pos_j in range(len(r_j) + 1):
                        r_j_new = r_j[:pos_j] + [c] + r_j[pos_j:]
                        # Check capacity
                        dem = sum(demands[x] for x in r_j_new if x <= n_customers)
                        if dem > capacity:
                            continue
                        feasible_j, d_j, _ = _insert_station(r_j_new, D, demands,
                                                              ready, due, service,
                                                              n_customers, stations,
                                                              battery_max,
                                                              recharge_rate, horizon, speed)
                        if feasible_j and d_j < best_j_d:
                            best_j_d = d_j
                            best_pos = pos_j
                    if best_pos != -1:
                        total_new = d_i + best_j_d
                        total_old = route_dists[i] + route_dists[j]
                        if total_new < total_old - 1e-6:
                            if r_i_without:
                                routes[i] = r_i_without
                                route_dists[i] = d_i
                            else:
                                routes[i] = []
                                route_dists[i] = 0.0
                            routes[j] = routes[j][:best_pos] + [c] + routes[j][best_pos:]
                            route_dists[j] = best_j_d
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break

        # Clean up empty routes
        if improved:
            new_routes = []
            for r in routes:
                if r:
                    new_routes.append(r)
            routes = new_routes

    # Final fallback: if any customers unvisited (shouldn't happen), add them
    visited = set()
    for r in routes:
        for c in r:
            if c <= n_customers:
                visited.add(c)
    for c in customers:
        if c not in visited:
            routes.append([c])

    return [r for r in routes if r]
# EVOLVE-BLOCK-END
