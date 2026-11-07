# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def plan(instance):
    start_time = time.time()
    deadline = start_time + 143.0  # 7s safety margin for 150s limit

    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    # Precompute distance matrix
    n_nodes = len(coords)
    dx = coords[:, 0]
    dy = coords[:, 1]
    dist = np.sqrt((dx[:, None] - dx[None, :])**2 + (dy[:, None] - dy[None, :])**2)

    customers = list(range(1, n_customers + 1))
    customer_set = set(customers)

    def simulate_route_fast(route):
        """
        Fast simulation of a route (list of node ids) starting from depot (node 0).
        Returns (total_distance, feasible) tuple. Feasible is True if all constraints met.
        """
        if not route:
            return (0.0, True)

        t = 0.0
        battery = float(battery_cap)
        total_dist = 0.0
        current = 0

        for node in route:
            d = float(dist[current, node])
            t += d / speed
            battery -= d
            total_dist += d

            if battery < -1e-9:
                return (total_dist, False)

            if node in customer_set:
                if t < ready[node]:
                    t = ready[node]
                if t > due[node] + 1e-9:
                    return (total_dist, False)
                t += service[node]
            elif node in stations:
                recharge_time = (battery_cap - battery) * recharge_rate
                t += recharge_time
                battery = float(battery_cap)

            current = node

        # Return to depot
        d = float(dist[current, 0])
        t += d / speed
        battery -= d
        total_dist += d

        if battery < -1e-9:
            return (total_dist, False)
        if t > horizon + 1e-9:
            return (total_dist, False)

        return (total_dist, True)

    def route_demand(route):
        return float(np.sum(demands[route])) if route else 0.0

    def build_route_savings(order):
        """
        Build routes using a savings-based approach.
        order is the initial ordering of customers.
        Returns list of routes or None if infeasible.
        """
        unvisited = set(customers)
        routes = []  # Each route is a list of node ids

        # Start with each customer as its own route
        for c in order:
            if c not in unvisited:
                continue
            r = [c]
            _, feasible = simulate_route_fast(r)
            if not feasible:
                return None
            routes.append(r)
            unvisited.remove(c)

        # Compute savings for merging pairs
        # Savings of merging route_i ending at a and route_j starting at b:
        # save = dist(0,a) + dist(0,b) - dist(a,b)
        # We iteratively merge the best feasible pair

        improved = True
        while improved and unvisited:
            improved = False
            best_save = 0.0
            best_i = -1
            best_j = -1
            best_pos = None  # Where to merge: (i, j, pos_in_i, pos_in_j)

            n_routes = len(routes)
            for i in range(n_routes):
                for j in range(i + 1, n_routes):
                    ri = routes[i]
                    rj = routes[j]

                    # Try all possible merge positions
                    # Merge: take a suffix of ri and prefix of rj, or vice versa
                    # Simplest: try appending rj to ri or ri to rj
                    # Or insert elements of rj into ri

                    # Option 1: append rj to ri
                    if route_demand(ri + rj) <= capacity:
                        merged = ri + rj
                        _, feas = simulate_route_fast(merged)
                        if feas:
                            save = float(dist[0, ri[-1]]) + float(dist[0, rj[0]]) - float(dist[ri[-1], rj[0]])
                            if save > best_save:
                                best_save = save
                                best_i = i
                                best_j = j
                                best_pos = ('append', i, j)

                    # Option 2: append ri to rj
                    if route_demand(rj + ri) <= capacity:
                        merged = rj + ri
                        _, feas = simulate_route_fast(merged)
                        if feas:
                            save = float(dist[0, rj[-1]]) + float(dist[0, ri[0]]) - float(dist[rj[-1], ri[0]])
                            if save > best_save:
                                best_save = save
                                best_i = i
                                best_j = j
                                best_pos = ('append_rev', i, j)

                    # Option 3: Insert each element of rj into ri
                    for k in range(len(rj)):
                        c = rj[k]
                        if route_demand(ri + [c]) > capacity:
                            continue
                        for pos in range(len(ri) + 1):
                            merged = ri[:pos] + [c] + ri[pos:]
                            _, feas = simulate_route_fast(merged)
                            if feas:
                                if pos == 0:
                                    save = float(dist[0, ri[0]]) + float(dist[0, c]) - float(dist[0, c]) - float(dist[c, ri[0]]) + float(dist[0, ri[0]])
                                    # Simpler: save = dist(0,ri[0]) - dist(c,ri[0]) if c inserted at front
                                    # Actually savings = dist(0,old_first) - dist(c,old_first) when inserting c before old_first
                                    # But we also remove c from its own route... this gets complex
                                    # Let's just use the distance reduction
                                    old_d = float(dist[0, ri[0]]) + float(dist[0, c])  # approximate
                                    new_d = float(dist[0, c]) + float(dist[c, ri[0]])
                                    save = old_d - new_d
                                elif pos == len(ri):
                                    old_d = float(dist[ri[-1], 0]) + float(dist[0, c])
                                    new_d = float(dist[ri[-1], c]) + float(dist[c, 0])
                                    save = old_d - new_d
                                else:
                                    old_d = float(dist[ri[pos-1], ri[pos]])
                                    new_d = float(dist[ri[pos-1], c]) + float(dist[c, ri[pos]])
                                    save = old_d - new_d

                                if save > best_save:
                                    best_save = save
                                    best_i = i
                                    best_j = j
                                    best_pos = ('insert', i, j, k, pos)

            if best_i >= 0 and best_pos is not None:
                if best_pos[0] == 'append':
                    _, _, i, j = best_pos
                    merged = routes[i] + routes[j]
                    routes[i] = merged
                    routes.pop(j)
                elif best_pos[0] == 'append_rev':
                    _, _, i, j = best_pos
                    merged = routes[j] + routes[i]
                    routes[i] = merged
                    routes.pop(j)
                elif best_pos[0] == 'insert':
                    _, _, i, j, k, pos = best_pos
                    c = routes[j][k]
                    routes[i] = routes[i][:pos] + [c] + routes[i][pos:]
                    routes[j].pop(k)
                    if not routes[j]:
                        routes.pop(j)
                        if j < i:
                            i -= 1
                improved = True

        # Check all customers are served
        all_served = []
        for r in routes:
            for c in r:
                if c in customer_set:
                    all_served.append(c)

        if len(set(all_served)) != len(customers):
            return None

        return routes

    def two_opt(route):
        """
        Apply 2-opt improvement to a single route.
        """
        improved = True
        max_iter = 50
        iterations = 0
        while improved and iterations < max_iter:
            improved = False
            iterations += 1
            n = len(route)
            for i in range(n - 1):
                for j in range(i + 1, n):
                    new_route = list(route[:i]) + list(reversed(route[i:j+1])) + list(route[j+1:])
                    if route_demand(new_route) > capacity:
                        continue
                    _, feas = simulate_route_fast(new_route)
                    if not feas:
                        continue
                    old_d, _ = simulate_route_fast(route)
                    new_d, _ = simulate_route_fast(new_route)
                    if new_d < old_d - 1e-6:
                        route = new_route
                        improved = True

        return route

    def or_opt(route):
        """
        Apply Or-opt (move 1-2 consecutive nodes) improvement to a single route.
        """
        improved = True
        max_iter = 30
        iterations = 0
        while improved and iterations < max_iter:
            improved = False
            iterations += 1
            n = len(route)
            for i in range(n):
                for length in [1, 2]:
                    if i + length > n:
                        continue
                    segment = route[i:i+length]
                    remaining = route[:i] + route[i+length:]
                    if not remaining:
                        continue
                    for pos in range(len(remaining) + 1):
                        new_route = remaining[:pos] + segment + remaining[pos:]
                        if route_demand(new_route) > capacity:
                            continue
                        _, feas = simulate_route_fast(new_route)
                        if not feas:
                            continue
                        old_d, _ = simulate_route_fast(route)
                        new_d, _ = simulate_route_fast(new_route)
                        if new_d < old_d - 1e-6:
                            route = new_route
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

        return route

    def relocate_customer(routes):
        """
        Try to move a customer from one route to another if it improves total distance.
        """
        improved = False
        n_routes = len(routes)

        for i in range(n_routes):
            ri = routes[i]
            for pos_i in range(len(ri)):
                c = ri[pos_i]
                new_ri = ri[:pos_i] + ri[pos_i+1:]
                old_di, feas_i = simulate_route_fast(new_ri) if new_ri else (0.0, True)
                if not feas_i:
                    continue

                for j in range(n_routes):
                    if i == j:
                        continue
                    rj = routes[j]
                    if route_demand(rj + [c]) > capacity:
                        continue
                    for pos_j in range(len(rj) + 1):
                        new_rj = rj[:pos_j] + [c] + rj[pos_j:]
                        old_dj, feas_j = simulate_route_fast(rj)
                        if not feas_j:
                            continue
                        new_dj, feas_new_j = simulate_route_fast(new_rj)
                        if not feas_new_j:
                            continue

                        old_total = old_di + old_dj
                        new_total = old_di + new_dj
                        if new_total < old_total - 1e-6:
                            routes[i] = new_ri
                            routes[j] = new_rj
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

    def swap_customers(routes):
        """
        Try swapping two customers between routes if it improves total distance.
        """
        improved = False
        n_routes = len(routes)

        for i in range(n_routes):
            ri = routes[i]
            for pos_i in range(len(ri)):
                ci = ri[pos_i]
                for j in range(i + 1, n_routes):
                    rj = routes[j]
                    for pos_j in range(len(rj)):
                        cj = rj[pos_j]
                        if demands[ci] == demands[cj]:
                            continue

                        new_ri = list(ri)
                        new_rj = list(rj)
                        new_ri[pos_i] = cj
                        new_rj[pos_j] = ci

                        if route_demand(new_ri) > capacity or route_demand(new_rj) > capacity:
                            continue

                        _, feas_i = simulate_route_fast(new_ri)
                        _, feas_j = simulate_route_fast(new_rj)
                        if not feas_i or not feas_j:
                            continue

                        old_di, _ = simulate_route_fast(ri)
                        old_dj, _ = simulate_route_fast(rj)
                        new_di, _ = simulate_route_fast(new_ri)
                        new_dj, _ = simulate_route_fast(new_rj)

                        if new_di + new_dj < old_di + old_dj - 1e-6:
                            routes[i] = new_ri
                            routes[j] = new_rj
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

        return routes, improved

    def total_distance(routes):
        total = 0.0
        for r in routes:
            d, feas = simulate_route_fast(r)
            if not feas:
                return float('inf')
            total += d
        return total

    # Initial construction
    # Try multiple orderings and pick the best
    best_routes = None
    best_dist = float('inf')

    # Order by due time
    order1 = sorted(customers, key=lambda v: due[v])
    routes1 = build_route_savings(order1)
    if routes1 is not None:
        # Apply local search
        for r in range(len(routes1)):
            routes1[r] = two_opt(routes1[r])
            routes1[r] = or_opt(routes1[r])
        d1 = total_distance(routes1)
        if d1 < best_dist:
            best_dist = d1
            best_routes = routes1

    # Order by x-coordinate (cluster-like)
    order2 = sorted(customers, key=lambda v: coords[v, 0])
    routes2 = build_route_savings(order2)
    if routes2 is not None:
        for r in range(len(routes2)):
            routes2[r] = two_opt(routes2[r])
            routes2[r] = or_opt(routes2[r])
        d2 = total_distance(routes2)
        if d2 < best_dist:
            best_dist = d2
            best_routes = routes2

    # Order by angle from depot
    order3 = sorted(customers, key=lambda v: math.atan2(coords[v, 1] - coords[0, 1], coords[v, 0] - coords[0, 0]))
    routes3 = build_route_savings(order3)
    if routes3 is not None:
        for r in range(len(routes3)):
            routes3[r] = two_opt(routes3[r])
            routes3[r] = or_opt(routes3[r])
        d3 = total_distance(routes3)
        if d3 < best_dist:
            best_dist = d3
            best_routes = routes3

    if best_routes is None:
        # Fallback: one customer per route
        best_routes = [[c] for c in customers]
        best_dist = total_distance(best_routes)

    # Iterative local search
    no_improve_count = 0
    max_no_improve = 10

    while time.time() < deadline - 5.0 and no_improve_count < max_no_improve:
        routes_copy = [list(r) for r in best_routes]
        old_dist = best_dist

        # Relocate
        routes_copy, imp1 = relocate_customer(routes_copy)
        if imp1:
            for r in range(len(routes_copy)):
                routes_copy[r] = two_opt(routes_copy[r])
                routes_copy[r] = or_opt(routes_copy[r])
            d = total_distance(routes_copy)
            if d < best_dist - 1e-6:
                best_routes = routes_copy
                best_dist = d
                no_improve_count = 0
                continue
            else:
                routes_copy = [list(r) for r in best_routes]

        # Swap
        routes_copy, imp2 = swap_customers(routes_copy)
        if imp2:
            for r in range(len(routes_copy)):
                routes_copy[r] = two_opt(routes_copy[r])
                routes_copy[r] = or_opt(routes_copy[r])
            d = total_distance(routes_copy)
            if d < best_dist - 1e-6:
                best_routes = routes_copy
                best_dist = d
                no_improve_count = 0
                continue
            else:
                routes_copy = [list(r) for r in best_routes]

        # 2-opt + Or-opt pass
        for r in range(len(routes_copy)):
            routes_copy[r] = two_opt(routes_copy[r])
            routes_copy[r] = or_opt(routes_copy[r])
        d = total_distance(routes_copy)
        if d < best_dist - 1e-6:
            best_routes = routes_copy
            best_dist = d
            no_improve_count = 0
        else:
            no_improve_count += 1

    # Final validation
    all_served = []
    for route in best_routes:
        for c in route:
            if c in customer_set:
                all_served.append(c)

    if len(all_served) != len(customers) or len(set(all_served)) != len(customers):
        best_routes = [[c] for c in customers]

    return best_routes
# EVOLVE-BLOCK-END
