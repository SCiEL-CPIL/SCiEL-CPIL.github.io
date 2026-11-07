"""EV routing: Clarke-Wright savings + aggressive local search with battery-aware feasibility."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random
import heapq
import copy


def plan(instance):
    start = time.time()
    deadline = start + 145.0

    coords = np.asarray(instance["coords"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    ready = np.asarray(instance["ready"], dtype=float)
    due = np.asarray(instance["due"], dtype=float)
    service = np.asarray(instance["service"], dtype=float)
    n_customers = instance["n_customers"]
    stations = list(instance["stations"])
    capacity = float(instance["capacity"])
    battery_cap = float(instance["battery"])
    speed = float(instance["speed"])
    recharge_rate = float(instance["recharge_rate"])
    horizon = float(instance["horizon"])

    n = coords.shape[0]
    diff = coords[:, None, :] - coords[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    dist = np.sqrt(d2)

    customers = list(range(1, n_customers + 1))
    n_cust = len(customers)

    # ---------- Fast battery-aware simulation ----------
    def simulate(route):
        """Check if route is feasible (capacity, battery, time windows, horizon). Returns (ok, total_dist)."""
        if not route:
            return True, 0.0
        load = sum(demands[v] for v in route)
        if load > capacity + 1e-9:
            return False, 0.0

        t = 0.0
        battery = battery_cap
        total = 0.0
        prev = 0

        for v in route:
            d = dist[prev, v]
            if d > battery + 1e-9:
                # Try to find a station to recharge
                charged = False
                for st in stations:
                    d_to_st = dist[prev, st]
                    d_st_to_v = dist[st, v]
                    if d_to_st <= battery + 1e-9:
                        battery -= d_to_st
                        t += d_to_st / speed
                        total += d_to_st
                        if battery < d_st_to_v:
                            charge_time = (d_st_to_v - battery) / recharge_rate
                            t += charge_time
                            battery = battery_cap
                        d = d_st_to_v
                        if d <= battery + 1e-9:
                            battery -= d
                            t += d / speed
                            total += d
                            charged = True
                            break
                if not charged:
                    return False, 0.0
            else:
                battery -= d
                t += d / speed
                total += d

            if t < ready[v]:
                t = ready[v]
            if t > due[v] + 1e-9:
                return False, 0.0
            t += service[v]
            prev = v

        # Return to depot
        d = dist[prev, 0]
        if d > battery + 1e-9:
            charged = False
            for st in stations:
                d_to_st = dist[prev, st]
                d_st_to_0 = dist[st, 0]
                if d_to_st <= battery + 1e-9:
                    battery -= d_to_st
                    t += d_to_st / speed
                    total += d_to_st
                    if battery < d_st_to_0:
                        charge_time = (d_st_to_0 - battery) / recharge_rate
                        t += charge_time
                        battery = battery_cap
                    d = d_st_to_0
                    if d <= battery + 1e-9:
                        battery -= d
                        t += d / speed
                        total += d
                        charged = True
                        break
            if not charged:
                return False, 0.0
        else:
            battery -= d
            t += d / speed
            total += d

        if t > horizon + 1e-9:
            return False, 0.0
        return True, total

    # ---------- Route distance (for optimization, ignoring charging detours) ----------
    def route_dist(route):
        if not route:
            return 0.0
        t = dist[0, route[0]]
        for i in range(len(route) - 1):
            t += dist[route[i], route[i + 1]]
        t += dist[route[-1], 0]
        return t

    def total_distance(routes):
        return sum(route_dist(r) for r in routes)

    # ---------- Clarke-Wright Savings Construction ----------
    def cw_construct():
        """Build routes using Clarke-Wright savings heuristic."""
        # Start with each customer in its own route
        routes = [[c] for c in customers]
        route_of = {c: i for i, c in enumerate(customers)}
        
        # Precompute savings: s(i,j) = d(0,i) + d(0,j) - d(i,j)
        savings = []
        for i in range(len(customers)):
            for j in range(i + 1, len(customers)):
                ci = customers[i]
                cj = customers[j]
                s = dist[0, ci] + dist[0, cj] - dist[ci, cj]
                savings.append((s, ci, cj))
        
        # Sort by savings descending
        savings.sort(reverse=True)
        
        for s, ci, cj in savings:
            if time.time() > deadline - 30.0:
                break
            if s <= 0:
                break
            ri = route_of[ci]
            rj = route_of[cj]
            if ri == rj:
                continue
            route_i = routes[ri]
            route_j = routes[rj]
            
            # Check capacity
            load_i = sum(demands[v] for v in route_i)
            load_j = sum(demands[v] for v in route_j)
            if load_i + load_j > capacity + 1e-9:
                continue
            
            # Check if ci and cj are at ends of their routes
            ci_at_end = (ci == route_i[0] or ci == route_i[-1])
            cj_at_end = (cj == route_j[0] or cj == route_j[-1])
            if not ci_at_end or not cj_at_end:
                continue
            
            # Determine merged route
            if ci == route_i[0] and cj == route_j[0]:
                new_route = route_i + route_j
            elif ci == route_i[-1] and cj == route_j[0]:
                new_route = route_i + route_j
            elif ci == route_i[0] and cj == route_j[-1]:
                new_route = route_i + route_j[::-1]
            else:
                new_route = route_i + route_j[::-1]
            
            # Check feasibility
            ok, _ = simulate(new_route)
            if ok:
                routes[ri] = new_route
                routes[rj] = []
                for v in new_route:
                    route_of[v] = ri
        
        # Remove empty routes
        routes = [r for r in routes if r]
        return routes

    # ---------- 2-opt ----------
    def two_opt(route, max_iter=500):
        if len(route) < 3:
            return route
        best = list(route)
        best_dist = route_dist(best)
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            L = len(best)
            for i in range(L - 1):
                a = best[i - 1] if i > 0 else 0
                b = best[i]
                for j in range(i + 1, L):
                    c = best[j]
                    d_node = best[j + 1] if j + 1 < L else 0
                    old = dist[a, b] + dist[c, d_node]
                    new = dist[a, c] + dist[b, d_node]
                    if new < old - 1e-9:
                        cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                        ok, _ = simulate(cand)
                        if ok and route_dist(cand) < best_dist - 1e-9:
                            best = cand
                            best_dist = route_dist(cand)
                            improved = True
                            break
                if improved:
                    break
        return best

    # ---------- Or-opt ----------
    def or_opt(route, max_iter=300):
        if len(route) < 3:
            return route
        best = list(route)
        best_dist = route_dist(best)
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            L = len(best)
            for seg_len in (1, 2):
                for i in range(L - seg_len + 1):
                    seg = best[i:i + seg_len]
                    rest = best[:i] + best[i + seg_len:]
                    if not rest:
                        continue
                    for pos in range(len(rest) + 1):
                        cand = rest[:pos] + seg + rest[pos:]
                        if cand == best:
                            continue
                        ok, _ = simulate(cand)
                        if ok and route_dist(cand) < best_dist - 1e-9:
                            best = cand
                            best_dist = route_dist(cand)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return best

    # ---------- Relocate between routes ----------
    def relocate_opt(routes, max_iter=50):
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            n_routes = len(routes)
            for r1 in range(n_routes):
                if len(routes[r1]) == 0:
                    continue
                for r2 in range(n_routes):
                    if r1 == r2:
                        continue
                    route1 = list(routes[r1])
                    route2 = list(routes[r2])
                    for i in range(len(route1)):
                        v = route1[i]
                        load2 = sum(demands[x] for x in route2)
                        if load2 + demands[v] > capacity + 1e-9:
                            continue
                        new_route1 = route1[:i] + route1[i+1:]
                        best_pos = -1
                        best_cost = float('inf')
                        for pos in range(len(route2) + 1):
                            cand = route2[:pos] + [v] + route2[pos:]
                            ok, _ = simulate(cand)
                            if ok:
                                cost = route_dist(cand)
                                if cost < best_cost:
                                    best_cost = cost
                                    best_pos = pos
                        if best_pos >= 0:
                            new_route2 = route2[:best_pos] + [v] + route2[best_pos:]
                            ok1, _ = simulate(new_route1)
                            if ok1:
                                old_total = route_dist(route1) + route_dist(route2)
                                new_total = route_dist(new_route1) + best_cost
                                if new_total < old_total - 1e-9:
                                    routes[r1] = new_route1
                                    routes[r2] = new_route2
                                    improved = True
                                    break
                    if improved:
                        break
                if improved:
                    break
        return routes

    # ---------- Swap between routes ----------
    def swap_opt(routes, max_iter=30):
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            n_routes = len(routes)
            for r1 in range(n_routes):
                for r2 in range(r1 + 1, n_routes):
                    route1 = list(routes[r1])
                    route2 = list(routes[r2])
                    for i in range(len(route1)):
                        v1 = route1[i]
                        for j in range(len(route2)):
                            v2 = route2[j]
                            load1 = sum(demands[x] for x in route1) - demands[v1] + demands[v2]
                            load2 = sum(demands[x] for x in route2) - demands[v2] + demands[v1]
                            if load1 > capacity + 1e-9 or load2 > capacity + 1e-9:
                                continue
                            new_route1 = route1[:i] + [v2] + route1[i+1:]
                            new_route2 = route2[:j] + [v1] + route2[j+1:]
                            ok1, _ = simulate(new_route1)
                            ok2, _ = simulate(new_route2)
                            if ok1 and ok2:
                                old_total = route_dist(route1) + route_dist(route2)
                                new_total = route_dist(new_route1) + route_dist(new_route2)
                                if new_total < old_total - 1e-9:
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

    # ---------- Main ----------
    # Phase 1: Clarke-Wright construction
    routes = cw_construct()

    # Phase 2: Local search
    for r in range(len(routes)):
        if time.time() > deadline - 20.0:
            break
        routes[r] = two_opt(routes[r])
        routes[r] = or_opt(routes[r])

    if time.time() < deadline - 25.0:
        routes = relocate_opt(routes)
    if time.time() < deadline - 20.0:
        routes = swap_opt(routes)

    # Phase 3: Second local search pass
    for r in range(len(routes)):
        if time.time() > deadline - 10.0:
            break
        routes[r] = two_opt(routes[r])
        routes[r] = or_opt(routes[r])

    if time.time() < deadline - 10.0:
        routes = relocate_opt(routes)

    # Phase 4: Final polish
    if time.time() < deadline - 5.0:
        for r in range(len(routes)):
            if time.time() > deadline - 3.0:
                break
            routes[r] = two_opt(routes[r])
            routes[r] = or_opt(routes[r])

    # Ensure all customers served
    if not routes:
        routes = [[v] for v in customers]
    else:
        served = set()
        for r in routes:
            for v in r:
                served.add(v)
        missing = [v for v in customers if v not in served]
        for v in missing:
            routes.append([v])

    return routes
# EVOLVE-BLOCK-END
