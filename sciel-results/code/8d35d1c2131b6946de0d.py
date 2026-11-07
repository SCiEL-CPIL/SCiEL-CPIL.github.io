"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random

def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    coords = np.array(instance["coords"], dtype=float)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    n_nodes = len(coords)
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = math.hypot(coords[i, 0] - coords[j, 0], coords[i, 1] - coords[j, 1])
            dist[i, j] = d
            dist[j, i] = d

    def check_route(route):
        if not route:
            return True, 0.0
        total_dist = 0.0
        t = 0.0
        bat = battery_cap
        for i, node in enumerate(route):
            prev = 0 if i == 0 else route[i - 1]
            d = dist[prev, node]
            total_dist += d
            bat -= d
            if bat < 0:
                return False, total_dist
            t += d / speed
            if node in stations:
                charge_time = (battery_cap - bat) * recharge_rate
                t += charge_time
                bat = battery_cap
            else:
                t = max(t, ready[node])
                if t > due[node]:
                    return False, total_dist
                t += service[node]
        d = dist[route[-1], 0]
        total_dist += d
        bat -= d
        if bat < 0:
            return False, total_dist
        t += d / speed
        if t > horizon:
            return False, total_dist
        return True, total_dist

    def total_distance(routes):
        total = 0.0
        for r in routes:
            if not r:
                continue
            total += dist[0, r[0]]
            for i in range(len(r) - 1):
                total += dist[r[i], r[i + 1]]
            total += dist[r[-1], 0]
        return total

    def route_demand(route):
        return sum(demands[n] for n in route if n not in stations)

    # 1. Cluster-First Construction
    customers = list(range(1, n_customers + 1))
    
    # Calculate angle from depot for each customer
    angles = {}
    for c in customers:
        dx = coords[c, 0] - coords[0, 0]
        dy = coords[c, 1] - coords[0, 1]
        angles[c] = math.atan2(dy, dx)
    
    # Sort by angle
    sorted_customers = sorted(customers, key=lambda c: angles[c])
    
    routes = []
    current_route = []
    current_demand = 0
    
    # Simple sweep: add customers in angular order until capacity is hit
    # Then start a new route. This creates spatially coherent clusters.
    for c in sorted_customers:
        if current_demand + demands[c] <= capacity:
            current_route.append(c)
            current_demand += demands[c]
        else:
            if current_route:
                routes.append(current_route)
            current_route = [c]
            current_demand = demands[c]
    if current_route:
        routes.append(current_route)

    # 2. Local Search: Or-opt and Relocate
    # We will iterate until no improvement or deadline.
    
    def or_opt(routes):
        improved = False
        for i in range(len(routes)):
            r = routes[i]
            n = len(r)
            if n < 2:
                continue
            for length in [1, 2, 3]:
                for a in range(n):
                    if a + length > n:
                        continue
                    segment = r[a:a+length]
                    rest = r[:a] + r[a+length:]
                    for b in range(len(rest) + 1):
                        if b == a or b == a + length:
                            continue
                        new_r = rest[:b] + segment + rest[b:]
                        if new_r == r:
                            continue
                        if route_demand(new_r) > capacity:
                            continue
                        feas, d_new = check_route(new_r)
                        if not feas:
                            continue
                        d_old = total_distance([r])
                        if d_new < d_old - 1e-9:
                            routes[i] = new_r
                            improved = True
                            # Restart this route's optimization
                            break
                    if improved:
                        break
                if improved:
                    break
        return improved

    def relocate(routes):
        improved = False
        for i in range(len(routes)):
            r_i = routes[i]
            for k in range(len(r_i)):
                c = r_i[k]
                r_i_new = r_i[:k] + r_i[k+1:]
                if not r_i_new:
                    continue
                feas_i, d_i = check_route(r_i_new)
                if not feas_i:
                    continue
                for j in range(len(routes)):
                    if i == j:
                        continue
                    r_j = routes[j]
                    for pos in range(len(r_j) + 1):
                        r_j_new = r_j[:pos] + [c] + r_j[pos:]
                        if route_demand(r_j_new) > capacity:
                            continue
                        feas_j, d_j = check_route(r_j_new)
                        if not feas_j:
                            continue
                        d_old = total_distance([r_i, r_j])
                        d_new = total_distance([r_i_new, r_j_new])
                        if d_new < d_old - 1e-9:
                            routes[i] = r_i_new
                            routes[j] = r_j_new
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        return improved

    # 3. Inter-route 2-opt (Swap tails of two routes)
    def inter_2opt(routes):
        improved = False
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                r_i = routes[i]
                r_j = routes[j]
                if not r_i or not r_j:
                    continue
                # Try swapping suffixes
                # r_i = [A], r_j = [B] -> r_i' = [A_tail, B_head], r_j' = [B_tail, A_head]
                # Actually, standard inter-route 2-opt reverses segments.
                # Simpler: try moving a prefix of one to the end of another?
                # Let's stick to Or-opt and Relocate for now, they are very effective.
                # But let's add a simple "swap" of two nodes if it helps.
                for k in range(len(r_i)):
                    for l in range(len(r_j)):
                        c1 = r_i[k]
                        c2 = r_j[l]
                        if c1 == c2: continue
                        # Swap
                        new_r_i = r_i.copy()
                        new_r_j = r_j.copy()
                        new_r_i[k] = c2
                        new_r_j[l] = c1
                        if route_demand(new_r_i) > capacity or route_demand(new_r_j) > capacity:
                            continue
                        feas_i, d_i = check_route(new_r_i)
                        if not feas_i: continue
                        feas_j, d_j = check_route(new_r_j)
                        if not feas_j: continue
                        
                        d_old = total_distance([r_i, r_j])
                        d_new = total_distance([new_r_i, new_r_j])
                        if d_new < d_old - 1e-9:
                            routes[i] = new_r_i
                            routes[j] = new_r_j
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        return improved

    # Iterative Improvement
    while time.time() < deadline:
        improved = False
        if or_opt(routes):
            improved = True
        if time.time() >= deadline: break
        if relocate(routes):
            improved = True
        if time.time() >= deadline: break
        if inter_2opt(routes):
            improved = True
        if time.time() >= deadline: break
        if not improved:
            # If stuck, try a random perturbation? Or just break.
            # Given the small size, if no improvement, we are likely at a local optimum.
            # Let's try a few random restarts of the local search phase if time permits?
            # No, better to just break to ensure we return valid routes.
            break

    routes = [r for r in routes if r]
    return routes
# EVOLVE-BLOCK-END
