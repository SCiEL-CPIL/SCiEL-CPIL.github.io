"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def plan(instance):
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s kill

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

    def simulate_route(route):
        """Simulate a route. Returns (feasible, total_distance, final_time, final_battery)."""
        if not route:
            return True, 0.0, 0.0, battery

        total_demand = sum(demands[v] for v in route)
        if total_demand > capacity:
            return False, 0.0, 0.0, 0.0

        t = 0.0
        bat = battery
        prev = 0
        total_dist = 0.0

        for v in route:
            d = dist[prev, v]
            bat -= d
            t += d / speed
            total_dist += d

            if bat < 0:
                return False, 0.0, t, bat

            if t < ready[v]:
                t = ready[v]
            if t > due[v]:
                return False, 0.0, t, bat
            t += service[v]
            prev = v

        d = dist[prev, 0]
        bat -= d
        t += d / speed
        total_dist += d

        if bat < 0:
            return False, 0.0, t, bat
        if t > horizon:
            return False, 0.0, t, bat

        return True, total_dist, t, bat

    def check_route(route):
        """Check if a route is feasible. Returns (feasible, total_distance)."""
        feasible, total_dist, _, _ = simulate_route(route)
        return feasible, total_dist

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

    def try_insert_charging(route):
        """Try to make a route feasible by inserting charging stations. Returns (feasible, new_route, total_distance)."""
        # First check if route is feasible as-is
        f, d = check_route(route)
        if f:
            return True, route, d

        # Find where battery fails
        bat = battery
        prev = 0
        fail_idx = -1
        for i, v in enumerate(route):
            d_leg = dist[prev, v]
            bat -= d_leg
            if bat < 0:
                fail_idx = i
                break
            prev = v

        if fail_idx == -1:
            # Battery fails on return to depot
            # Try inserting station before last customer
            for st in stations:
                new_route = route[:-1] + [st] + route[-1:]
                f2, d2 = check_route(new_route)
                if f2:
                    return True, new_route, d2
            return False, route, 0.0

        # Insert station before route[fail_idx]
        if fail_idx == 0:
            prev_node = 0
        else:
            prev_node = route[fail_idx - 1]

        # Try each station
        best = None
        best_d = float('inf')
        for st in stations:
            d_to_st = dist[prev_node, st]
            # Check if we can reach station
            bat_to_st = battery
            t_to_st = 0.0
            p = 0
            for j in range(fail_idx):
                dd = dist[p, route[j]]
                bat_to_st -= dd
                t_to_st += dd / speed
                if t_to_st < ready[route[j]]:
                    t_to_st = ready[route[j]]
                if t_to_st > due[route[j]]:
                    break
                t_to_st += service[route[j]]
                p = route[j]
            else:
                bat_to_st -= d_to_st
                if bat_to_st >= 0:
                    new_route = route[:fail_idx] + [st] + route[fail_idx:]
                    f2, d2 = check_route(new_route)
                    if f2 and d2 < best_d:
                        best_d = d2
                        best = new_route

        if best is not None:
            return True, best, best_d

        # Try two stations
        for i in range(len(route) + 1):
            for j in range(i, len(route) + 1):
                for st1 in stations:
                    for st2 in stations:
                        new_route = route[:i] + [st1] + route[i:j] + [st2] + route[j:]
                        f2, d2 = check_route(new_route)
                        if f2 and d2 < best_d:
                            best_d = d2
                            best = new_route

        if best is not None:
            return True, best, best_d
        return False, route, 0.0

    def savings_construct():
        """Clarke-Wright savings based construction."""
        routes = [[v] for v in customers]
        route_of = {}
        for i, v in enumerate(customers):
            route_of[v] = i

        savings = []
        for i in range(n_customers):
            for j in range(i + 1, n_customers):
                vi, vj = i + 1, j + 1
                s = dist[0, vi] + dist[0, vj] - dist[vi, vj]
                savings.append((s, vi, vj))

        savings.sort(reverse=True)

        for s, vi, vj in savings:
            if time.time() > deadline:
                break
            ri = route_of[vi]
            rj = route_of[vj]
            if ri == rj:
                continue

            route_i = routes[ri]
            route_j = routes[rj]

            can_merge = False
            new_route = None

            if route_i[-1] == vi and route_j[0] == vj:
                candidate = route_i + route_j
                if sum(demands[x] for x in candidate) <= capacity:
                    f, _ = check_route(candidate)
                    if f:
                        can_merge = True
                        new_route = candidate
            elif route_j[-1] == vj and route_i[0] == vi:
                candidate = route_j + route_i
                if sum(demands[x] for x in candidate) <= capacity:
                    f, _ = check_route(candidate)
                    if f:
                        can_merge = True
                        new_route = candidate
            elif route_i[0] == vi and route_j[0] == vj:
                candidate = route_i[::-1] + route_j
                if sum(demands[x] for x in candidate) <= capacity:
                    f, _ = check_route(candidate)
                    if f:
                        can_merge = True
                        new_route = candidate
            elif route_i[-1] == vi and route_j[-1] == vj:
                candidate = route_i + route_j[::-1]
                if sum(demands[x] for x in candidate) <= capacity:
                    f, _ = check_route(candidate)
                    if f:
                        can_merge = True
                        new_route = candidate

            if can_merge:
                routes[ri] = new_route
                routes[rj] = []
                for v in new_route:
                    route_of[v] = ri
                routes = [r for r in routes if r]
                for i, r in enumerate(routes):
                    for v in r:
                        route_of[v] = i

        return [r for r in routes if r]

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

                    if bat_after < 0:
                        continue
                    if load + demands[v] > capacity:
                        continue

                    if t_after < ready[v]:
                        t_after = ready[v]
                    if t_after > due[v]:
                        continue

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

    # Initial solution
    best_routes = [[v] for v in customers]
    best_distance = sum(dist[0, v] * 2 for v in customers)

    # Try savings construction
    if time.time() < deadline:
        sw_routes = savings_construct()
        if validate_routes(sw_routes):
            d = total_route_distance(sw_routes)
            if d < best_distance:
                best_routes = sw_routes
                best_distance = d

    # Try greedy construction
    orderings = [
        sorted(customers, key=lambda v: due[v]),
        sorted(customers, key=lambda v: ready[v]),
        sorted(customers, key=lambda v: dist[0, v]),
        sorted(customers, key=lambda v: (due[v], dist[0, v])),
    ]

    for ordering in orderings:
        if time.time() > deadline:
            break
        greedy_routes = greedy_construct(ordering)
        if validate_routes(greedy_routes):
            d = total_route_distance(greedy_routes)
            if d < best_distance:
                best_routes = greedy_routes
                best_distance = d

    # Fast feasibility check for 2-opt: only check affected segment
    def is_feasible_after_2opt(route, a, b):
        """
        Quick check: after reversing route[a:b+1], is the route still feasible?
        """
        new_route = route[:a] + route[a:b + 1][::-1] + route[b + 1:]
        f, _ = check_route(new_route)
        return f, new_route

    def two_opt_route(route):
        """Apply 2-opt to a single route until no improvement. Fast incremental distance."""
        if len(route) < 2:
            return route
        n = len(route)

        improved = True
        while improved and time.time() < deadline:
            improved = False
            for a in range(n - 1):
                if time.time() > deadline:
                    break
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
                        feasible, new_route = is_feasible_after_2opt(route, a, b)
                        if feasible:
                            route = new_route
                            n = len(route)
                            improved = True
                            break
                if improved:
                    break
        return route

    def or_opt_route(route):
        """Apply Or-opt moves to improve route."""
        if len(route) < 2:
            return route
        n = len(route)

        def route_dist(r):
            if not r:
                return 0.0
            d = dist[0, r[0]]
            for i in range(len(r) - 1):
                d += dist[r[i], r[i + 1]]
            d += dist[r[-1], 0]
            return d

        improved = True
        while improved and time.time() < deadline:
            improved = False
            n = len(route)
            cur_d = route_dist(route)
            for a in range(n):
                if time.time() > deadline:
                    break
                for seg_len in [1, 2, 3]:
                    if a + seg_len > n:
                        continue
                    segment = route[a:a + seg_len]
                    remaining = route[:a] + route[a + seg_len:]
                    if not remaining:
                        continue
                    for pos in range(len(remaining) + 1):
                        if pos == a or pos == a + seg_len:
                            continue
                        new_route = remaining[:pos] + segment + remaining[pos:]
                        feasible, _ = check_route(new_route)
                        if feasible:
                            new_d = route_dist(new_route)
                            if new_d < cur_d - 1e-6:
                                route = new_route
                                cur_d = new_d
                                n = len(route)
                                improved = True
                                break
                    if improved:
                        break
                if improved:
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
                    route_i_without = routes[i][:j] + routes[i][j + 1:]
                    if not route_i_without:
                        new_routes = [r for idx, r in enumerate(routes) if idx != i]
                    else:
                        new_routes = list(routes)
                        new_routes[i] = route_i_without

                    if route_i_without:
                        feasible_i, _ = check_route(route_i_without)
                        if not feasible_i:
                            continue

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

    # Apply 2-opt to all routes
    if time.time() < deadline:
        for i in range(len(best_routes)):
            if time.time() > deadline:
                break
            best_routes[i] = two_opt_route(best_routes[i])

    # Apply relocate search
    if time.time() < deadline:
        best_routes = relocate_search(best_routes, max_passes=5)

    # Cross-route swap local search (fixed distance calc)
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
                            if time.time() > deadline:
                                break
                            vi = best_routes[i][a]
                            vj = best_routes[j][b]
                            if vi == vj:
                                continue
                            new_route_i = best_routes[i][:]
                            new_route_j = best_routes[j][:]
                            new_route_i[a] = vj
                            new_route_j[b] = vi

                            f_i, _ = check_route(new_route_i)
                            if not f_i:
                                continue
                            f_j, _ = check_route(new_route_j)
                            if not f_j:
                                continue

                            # Compute new distance for both routes
                            def rd(r):
                                if not r:
                                    return 0.0
                                d = dist[0, r[0]]
                                for k in range(len(r) - 1):
                                    d += dist[r[k], r[k + 1]]
                                d += dist[r[-1], 0]
                                return d

                            new_dist = current_dist - rd(best_routes[i]) - rd(best_routes[j]) + rd(new_route_i) + rd(new_route_j)

                            if new_dist < current_dist - 1e-6:
                                best_routes[i] = new_route_i
                                best_routes[j] = new_route_j
                                current_dist = new_dist
                                improved_swap = True
                                break
                        if improved_swap:
                            break
                    if improved_swap:
                        break
                if improved_swap:
                    break

    # Final polish with 3-opt
    def three_opt_route(route):
        """Apply 3-opt (reverse segment) to a single route."""
        if len(route) < 3:
            return route
        n = len(route)

        def route_dist(r):
            if not r:
                return 0.0
            d = dist[0, r[0]]
            for k in range(len(r) - 1):
                d += dist[r[k], r[k + 1]]
            d += dist[r[-1], 0]
            return d

        improved = True
        while improved and time.time() < deadline:
            improved = False
            n = len(route)
            cur_d = route_dist(route)
            for a in range(n - 1):
                if time.time() > deadline:
                    break
                for b in range(a + 1, n):
                    if time.time() > deadline:
                        break
                    # Reverse route[a:b+1]
                    new_route = route[:a] + route[a:b + 1][::-1] + route[b + 1:]
                    feasible, _ = check_route(new_route)
                    if feasible:
                        new_d = route_dist(new_route)
                        if new_d < cur_d - 1e-6:
                            route = new_route
                            cur_d = new_d
                            n = len(route)
                            improved = True
                            break
            if improved:
                continue
            break
        return route

    if time.time() < deadline:
        for i in range(len(best_routes)):
            if time.time() > deadline:
                break
            best_routes[i] = two_opt_route(best_routes[i])

        if time.time() < deadline:
            for i in range(len(best_routes)):
                if time.time() > deadline:
                    break
                best_routes[i] = or_opt_route(best_routes[i])

        if time.time() < deadline:
            best_routes = relocate_search(best_routes, max_passes=3)

        if time.time() < deadline:
            for i in range(len(best_routes)):
                if time.time() > deadline:
                    break
                best_routes[i] = two_opt_route(best_routes[i])

        if time.time() < deadline:
            for i in range(len(best_routes)):
                if time.time() > deadline:
                    break
                best_routes[i] = three_opt_route(best_routes[i])

    # Remove empty routes
    best_routes = [r for r in best_routes if r]

    # Final validation: ensure all routes are feasible
    # If any route fails battery, try inserting charging stations
    final_routes = []
    for route in best_routes:
        f, _ = check_route(route)
        if f:
            final_routes.append(route)
        else:
            # Try to fix by inserting charging stations
            fixed = False
            # Try inserting each station at each position
            for st in stations:
                for pos in range(len(route) + 1):
                    new_route = route[:pos] + [st] + route[pos:]
                    f2, _ = check_route(new_route)
                    if f2:
                        final_routes.append(new_route)
                        fixed = True
                        break
                if fixed:
                    break
            if not fixed:
                # Try two stations
                for st1 in stations:
                    for st2 in stations:
                        for pos1 in range(len(route) + 1):
                            for pos2 in range(pos1 + 1, len(route) + 2):
                                mid = route[:pos1] + [st1] + route[pos1:]
                                insert_pos = pos2 if pos2 <= pos1 else pos2
                                new_route = mid[:insert_pos] + [st2] + mid[insert_pos:]
                                f2, _ = check_route(new_route)
                                if f2:
                                    final_routes.append(new_route)
                                    fixed = True
                                    break
                            if fixed:
                                break
                        if fixed:
                            break
                    if fixed:
                        break
            if not fixed:
                # Last resort: split the route
                mid = len(route) // 2
                part1 = route[:mid]
                part2 = route[mid:]
                if part1:
                    final_routes.append(part1)
                if part2:
                    final_routes.append(part2)

    # Final check: ensure all customers are covered
    all_served = []
    for r in final_routes:
        all_served.extend(r)
    if set(all_served) != set(customers) or len(all_served) != len(customers):
        # Fallback: use the original best_routes
        final_routes = [r for r in best_routes if r]

    return final_routes
# EVOLVE-BLOCK-END
