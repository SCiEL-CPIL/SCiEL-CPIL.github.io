# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def plan(instance):
    t0 = time.time()
    deadline = t0 + 140.0

    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_customers = int(instance["n_customers"])
    stations = list(instance["stations"])
    capacity = float(instance["capacity"])
    battery_cap = float(instance["battery"])
    speed = float(instance["speed"])
    recharge_rate = float(instance["recharge_rate"])
    horizon = float(instance["horizon"])

    n_nodes = len(coords)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff * diff).sum(-1))

    customers = list(range(1, n_customers + 1))
    cust_set = set(customers)
    station_set = set(stations)
    is_customer = np.zeros(n_nodes, dtype=bool)
    is_station = np.zeros(n_nodes, dtype=bool)
    for c in customers:
        is_customer[c] = True
    for s in stations:
        is_station[s] = True

    cust_demand = np.zeros(n_nodes)
    for c in customers:
        cust_demand[c] = demands[c]

    def simulate_fast(route):
        if not route:
            return (0.0, 0.0, battery_cap)
        t = 0.0
        bat = battery_cap
        cd = 0.0
        cur = 0
        for node in route:
            d = dist[cur, node]
            t += d / speed
            bat -= d
            if bat < -1e-9:
                return None
            cd += d
            if is_customer[node]:
                if t < ready[node]:
                    t = ready[node]
                if t > due[node] + 1e-9:
                    return None
                t += service[node]
            elif is_station[node]:
                t += (battery_cap - bat) * recharge_rate
                bat = battery_cap
            cur = node
        d = dist[cur, 0]
        t += d / speed
        bat -= d
        cd += d
        if bat < -1e-9 or t > horizon + 1e-9:
            return None
        return (cd, t, bat)

    def route_demand(route):
        s = 0.0
        for v in route:
            if is_customer[v]:
                s += cust_demand[v]
        return s

    def total_dist(routes):
        tot = 0.0
        for r in routes:
            s = simulate_fast(r)
            if s is None:
                return float("inf")
            tot += s[0]
        return tot

    def best_station_fix(route):
        """Insert cheapest single station if needed to make route feasible."""
        s = simulate_fast(route)
        if s is not None:
            return list(route), s[0]
        
        # Find where battery goes negative
        t = 0.0
        bat = battery_cap
        cur = 0
        neg_pos = -1
        for idx, node in enumerate(route):
            d = dist[cur, node]
            t += d / speed
            bat -= d
            if bat < -1e-9 and neg_pos == -1:
                neg_pos = idx
            if is_customer[node]:
                if t < ready[node]:
                    t = ready[node]
                t += service[node]
            elif is_station[node]:
                t += (battery_cap - bat) * recharge_rate
                bat = battery_cap
            cur = node
        if neg_pos == -1:
            d = dist[cur, 0]
            bat -= d
            if bat < -1e-9:
                neg_pos = len(route)
            else:
                return None, None

        best_route = None
        best_d = float('inf')
        # Search near the negative battery point first
        search_start = max(0, neg_pos - 2)
        search_end = min(len(route) + 1, neg_pos + 3)
        for p in range(search_start, search_end + 1):
            for st in stations:
                cand = list(route[:p]) + [st] + list(route[p:])
                if route_demand(cand) > capacity:
                    continue
                sc = simulate_fast(cand)
                if sc is not None:
                    if sc[0] < best_d:
                        best_route, best_d = cand, sc[0]
        
        # If not found, search all positions
        if best_route is None:
            for p in range(len(route) + 1):
                for st in stations:
                    cand = list(route[:p]) + [st] + list(route[p:])
                    if route_demand(cand) > capacity:
                        continue
                    sc = simulate_fast(cand)
                    if sc is not None:
                        if sc[0] < best_d:
                            best_route, best_d = cand, sc[0]
        
        if best_route is not None:
            return best_route, best_d
        return None, None

    def greedy_construct(order):
        routes = []
        for c in order:
            placed = False
            best = None
            best_cd = float('inf')
            for ri, r in enumerate(routes):
                if route_demand(r) + demands[c] > capacity:
                    continue
                # Try inserting c at various positions, prefer lowest distance delta
                positions = []
                for pos in range(len(r) + 1):
                    prev = r[pos - 1] if pos > 0 else 0
                    nxt = r[pos] if pos < len(r) else 0
                    delta = dist[prev, c] + dist[c, nxt] - dist[prev, nxt]
                    positions.append((delta, pos))
                positions.sort()
                
                for delta, pos in positions:
                    cand = list(r[:pos]) + [c] + list(r[pos:])
                    fixed, cd = best_station_fix(cand)
                    if fixed is not None:
                        if cd < best_cd:
                            best = (ri, fixed, cd)
                            best_cd = cd
                        break
            if best is not None:
                routes[best[0]] = best[1]
                placed = True
            if not placed:
                fixed, cd = best_station_fix([c])
                if fixed is None:
                    fixed = [c]
                routes.append(fixed)
        return routes

    def two_opt(route, max_iter=200, tol=1e-7):
        if len(route) < 2:
            return route
        cur = simulate_fast(route)
        if cur is None:
            return route
        cur_d = cur[0]
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            n = len(route)
            for i in range(n - 1):
                for j in range(i + 1, n):
                    a = route[i - 1] if i > 0 else 0
                    b = route[i]
                    c_node = route[j]
                    d_node = route[j + 1] if j + 1 < n else 0
                    old_len = dist[a, b] + dist[c_node, d_node]
                    new_len = dist[a, c_node] + dist[b, d_node]
                    if new_len >= old_len - tol:
                        continue
                    nr = list(route[:i]) + list(reversed(route[i:j + 1])) + list(route[j + 1:])
                    if route_demand(nr) > capacity:
                        continue
                    sn = simulate_fast(nr)
                    if sn is None:
                        continue
                    if sn[0] < cur_d - tol:
                        route = nr
                        cur_d = sn[0]
                        improved = True
            if time.time() > deadline - 5.0:
                break
        return route

    def or_opt(route, max_iter=50):
        """Or-opt: remove 1-3 consecutive nodes and reinsert elsewhere."""
        if len(route) < 3:
            return route
        cur = simulate_fast(route)
        if cur is None:
            return route
        cur_d = cur[0]
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            n = len(route)
            for length in [1, 2, 3]:
                for i in range(n - length + 1):
                    segment = route[i:i + length]
                    rest = route[:i] + route[i + length:]
                    if route_demand(rest) > capacity:
                        continue
                    # Try inserting segment at various positions in rest
                    best_pos = -1
                    best_new = None
                    best_dist = cur_d
                    for pos in range(len(rest) + 1):
                        cand = rest[:pos] + segment + rest[pos:]
                        if route_demand(cand) > capacity:
                            continue
                        sn = simulate_fast(cand)
                        if sn is not None and sn[0] < best_dist - 1e-7:
                            best_dist = sn[0]
                            best_pos = pos
                            best_new = cand
                    if best_new is not None:
                        route = best_new
                        cur_d = best_dist
                        improved = True
            if time.time() > deadline - 5.0:
                break
        return route

    def relocate_move(routes):
        """Move a customer from one route to another if it improves total distance."""
        improved = False
        n_routes = len(routes)
        for i in range(n_routes):
            if not routes[i]:
                continue
            for pos_i in range(len(routes[i])):
                c = routes[i][pos_i]
                new_route_i = routes[i][:pos_i] + routes[i][pos_i + 1:]
                sim_i = simulate_fast(new_route_i) if new_route_i else (0.0, 0.0, battery_cap)
                if sim_i is None:
                    continue
                old_dist_i = simulate_fast(routes[i])[0]
                for j in range(n_routes):
                    if i == j or not routes[j]:
                        continue
                    old_dist_j = simulate_fast(routes[j])[0]
                    # Try inserting c at various positions in route_j
                    for pos_j in range(len(routes[j]) + 1):
                        new_route_j = routes[j][:pos_j] + [c] + routes[j][pos_j:]
                        if route_demand(new_route_j) > capacity:
                            continue
                        sim_j = simulate_fast(new_route_j)
                        if sim_j is None:
                            continue
                        new_total = sim_i[0] + sim_j[0]
                        old_total = old_dist_i + old_dist_j
                        if new_total < old_total - 1e-6:
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
        # Remove empty routes
        routes = [r for r in routes if r]
        return routes, improved

    def swap_move(routes):
        """Swap two customers between routes if it improves total distance."""
        improved = False
        n_routes = len(routes)
        for i in range(n_routes):
            if not routes[i]:
                continue
            for j in range(i + 1, n_routes):
                if not routes[j]:
                    continue
                for pos_i in range(len(routes[i])):
                    for pos_j in range(len(routes[j])):
                        c_i = routes[i][pos_i]
                        c_j = routes[j][pos_j]
                        new_route_i = routes[i][:]
                        new_route_j = routes[j][:]
                        new_route_i[pos_i] = c_j
                        new_route_j[pos_j] = c_i
                        if route_demand(new_route_i) > capacity or route_demand(new_route_j) > capacity:
                            continue
                        sim_i = simulate_fast(new_route_i)
                        sim_j = simulate_fast(new_route_j)
                        if sim_i is None or sim_j is None:
                            continue
                        old_dist = simulate_fast(routes[i])[0] + simulate_fast(routes[j])[0]
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

    def two_opt_all_routes(routes, max_iter_per_route=50):
        for i in range(len(routes)):
            routes[i] = two_opt(routes[i], max_iter_per_route)
        return routes

    def or_opt_all_routes(routes, max_iter_per_route=20):
        for i in range(len(routes)):
            routes[i] = or_opt(routes[i], max_iter_per_route)
        return routes

    # Initial construction: sort by due time
    initial_order = sorted(customers, key=lambda v: due[v])
    routes = greedy_construct(initial_order)
    if routes is None:
        routes = [[c] for c in customers]

    best_routes = [list(r) for r in routes]
    best_dist = total_dist(best_routes)

    # Local search: 2-opt, or-opt, relocate, swap
    for iteration in range(300):
        if time.time() > deadline - 5.0:
            break

        # 2-opt
        routes_copy = [list(r) for r in best_routes]
        routes_copy = two_opt_all_routes(routes_copy, 30)
        d = total_dist(routes_copy)
        if d < best_dist - 1e-6:
            best_routes = routes_copy
            best_dist = d
            continue

        # Or-opt
        routes_copy = [list(r) for r in best_routes]
        routes_copy = or_opt_all_routes(routes_copy, 15)
        d = total_dist(routes_copy)
        if d < best_dist - 1e-6:
            best_routes = routes_copy
            best_dist = d
            continue

        # Relocate
        routes_copy = [list(r) for r in best_routes]
        routes_copy, imp1 = relocate_move(routes_copy)
        if imp1:
            d = total_dist(routes_copy)
            if d < best_dist - 1e-6:
                best_routes = routes_copy
                best_dist = d
                continue

        # Swap
        routes_copy = [list(r) for r in best_routes]
        routes_copy, imp2 = swap_move(routes_copy)
        if imp2:
            d = total_dist(routes_copy)
            if d < best_dist - 1e-6:
                best_routes = routes_copy
                best_dist = d
                continue

        # No improvement in this round, try again with different order
        if iteration % 10 == 0:
            random.shuffle(customers)
            routes2 = greedy_construct(customers[:])
            if routes2 is not None:
                routes2 = two_opt_all_routes(routes2, 20)
                d2 = total_dist(routes2)
                if d2 < best_dist - 1e-6:
                    best_routes = routes2
                    best_dist = d2

    # Final check: ensure all customers are served exactly once
    all_served = []
    for route in best_routes:
        for c in route:
            if c in customers:
                all_served.append(c)

    if len(all_served) != len(customers) or len(set(all_served)) != len(customers):
        best_routes = [[c] for c in customers]

    return best_routes
# EVOLVE-BLOCK-END
