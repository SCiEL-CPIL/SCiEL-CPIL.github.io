# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def _dist_matrix(coords):
    n = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            d = math.sqrt(dx * dx + dy * dy)
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _check_route(route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon):
    """Check if a route is feasible. Returns (feasible, total_distance, total_time)."""
    if not route:
        return True, 0.0, 0.0
    
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    
    total_demand = sum(demands[v] for v in route if v <= n_customers)
    if total_demand > capacity:
        return False, 0.0, 0.0
    
    battery = battery_max
    time_t = 0.0
    total_dist = 0.0
    prev = 0  # depot
    
    for v in route:
        d = dist[prev][v]
        battery -= d
        time_t += d / speed
        total_dist += d
        
        if battery < -1e-9:
            return False, 0.0, 0.0
        
        if v <= n_customers:
            if time_t < ready[v]:
                time_t = ready[v]
            if time_t > due[v] + 1e-9:
                return False, 0.0, 0.0
            time_t += service[v]
        else:
            energy_needed = battery_max - battery
            time_t += energy_needed * recharge_rate
            battery = battery_max
        
        prev = v
    
    d = dist[prev][0]
    battery -= d
    time_t += d / speed
    total_dist += d
    
    if battery < -1e-9:
        return False, 0.0, 0.0
    if time_t > horizon + 1e-9:
        return False, 0.0, 0.0
    
    return True, total_dist, time_t


def _route_distance(route, dist):
    """Quick distance calculation for a route."""
    if not route:
        return 0.0
    total = 0.0
    prev = 0
    for v in route:
        total += dist[prev][v]
        prev = v
    total += dist[prev][0]
    return total


def _construct_savings_routes(instance, dist, time_limit=5):
    """
    Construct initial routes using Clarke-Wright Savings Heuristic.
    This creates routes by merging nodes that save the most distance.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
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
    stations = instance["stations"]
    
    # Initialize routes: each customer is its own route
    routes = [[c] for c in range(1, n_customers + 1)]
    route_idx = {c: i for i, c in enumerate(range(1, n_customers + 1))}
    
    # Calculate savings for all pairs
    savings = []
    for i in range(1, n_customers + 1):
        for j in range(i + 1, n_customers + 1):
            s = dist[0][i] + dist[0][j] - dist[i][j]
            if s > 0:
                savings.append((s, i, j))
    
    savings.sort(reverse=True)
    
    # Merge routes
    for s, i, j in savings:
        if time.time() > deadline:
            break
        
        ri = route_idx.get(i)
        rj = route_idx.get(j)
        
        if ri is None or rj is None or ri == rj:
            continue
        
        # Check if i and j are at the ends of their respective routes
        route_i = routes[ri]
        route_j = routes[rj]
        
        # i must be at the end of route_i, j at the start of route_j (or vice versa)
        # Because routes are undirected in this representation, we need to check both ends.
        # Let's say we merge route_i and route_j.
        # Possible connections:
        # 1. route_i ends with i, route_j starts with j -> new route: route_i + route_j
        # 2. route_i ends with i, route_j ends with j -> new route: route_i + route_j[::-1]
        # 3. route_i starts with i, route_j starts with j -> new route: route_i[::-1] + route_j
        # 4. route_i starts with i, route_j ends with j -> new route: route_i[::-1] + route_j[::-1]
        
        possible_merges = []
        
        if route_i[-1] == i:
            if route_j[0] == j:
                possible_merges.append(route_i + route_j)
            if route_j[-1] == j:
                possible_merges.append(route_i + route_j[::-1])
        
        if route_i[0] == i:
            if route_j[0] == j:
                possible_merges.append(route_i[::-1] + route_j)
            if route_j[-1] == j:
                possible_merges.append(route_i[::-1] + route_j[::-1])
        
        best_merge = None
        best_dist = float('inf')
        
        for merge in possible_merges:
            # Check capacity
            total_demand = sum(demands[v] for v in merge if v <= n_customers)
            if total_demand > capacity:
                continue
            
            # Check feasibility
            feasible, d, t = _check_route(merge, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
            if feasible and d < best_dist:
                best_dist = d
                best_merge = merge
        
        if best_merge:
            # Perform merge
            routes[ri] = best_merge
            del routes[rj]
            
            # Update indices
            # This is inefficient, but for 25 customers it's fine.
            # We need to rebuild route_idx
            route_idx = {}
            for idx, r in enumerate(routes):
                for node in r:
                    route_idx[node] = idx
    
    return routes


def _local_search(routes, dist, instance, time_limit=40):
    """
    Efficient local search with 2-Opt, Or-Opt, and Relocate moves.
    Uses fast distance delta calculations to avoid full route re-evaluation
    unless a potential improvement is found.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    n_customers = instance["n_customers"]
    battery_max = instance["battery"]
    capacity = instance["capacity"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    # Precompute route distances
    route_dists = [_route_distance(r, dist) for r in routes]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. 2-Opt within routes
        for i in range(len(routes)):
            r = routes[i]
            n = len(r)
            if n < 2:
                continue
            
            for a in range(n - 1):
                for b in range(a + 1, n):
                    # Calculate delta distance for reversing r[a:b+1]
                    prev_a = r[a-1] if a > 0 else 0
                    next_b = r[b+1] if b < n - 1 else 0
                    
                    old_d = dist[prev_a][r[a]] + dist[r[b]][next_b]
                    new_d = dist[prev_a][r[b]] + dist[r[a]][next_b]
                    
                    if new_d < old_d - 1e-9:
                        new_r = r[:a] + r[a:b+1][::-1] + r[b+1:]
                        feasible, d, t = _check_route(new_r, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                        if feasible:
                            routes[i] = new_r
                            route_dists[i] = d
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            continue
            
        # 2. Or-Opt within routes (move 1 or 2 consecutive nodes)
        for i in range(len(routes)):
            r = routes[i]
            n = len(r)
            if n < 3:
                continue
            
            for L in [1, 2]:
                if n < L + 1:
                    continue
                for start in range(n - L):
                    moving = r[start:start+L]
                    if any(m > n_customers for m in moving):
                        continue
                    
                    rest = r[:start] + r[start+L:]
                    best_pos = -1
                    best_dist = float('inf')
                    
                    for pos in range(len(rest) + 1):
                        if pos == start or (start > 0 and pos == start + L) or (start == 0 and pos == L):
                            continue
                        
                        new_r = rest[:pos] + moving + rest[pos:]
                        d_new = _route_distance(new_r, dist)
                        if d_new < route_dists[i] - 1e-9:
                            feasible, d, t = _check_route(new_r, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                            if feasible:
                                best_pos = pos
                                best_dist = d
                                break
                    
                    if best_pos != -1:
                        new_r = rest[:best_pos] + moving + rest[best_pos:]
                        routes[i] = new_r
                        route_dists[i] = best_dist
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            continue

        # 3. Relocate: Move customer from route i to route j
        for i in range(len(routes)):
            r_i = routes[i]
            if not r_i:
                continue
            for pos_i in range(len(r_i)):
                c = r_i[pos_i]
                if c > n_customers:
                    continue
                
                for j in range(len(routes)):
                    if i == j:
                        continue
                    r_j = routes[j]
                    
                    demand_i = sum(instance["demands"][v] for v in r_i if v <= n_customers)
                    demand_j = sum(instance["demands"][v] for v in r_j if v <= n_customers)
                    
                    if demand_j + instance["demands"][c] > capacity:
                        continue
                    
                    best_pos_j = -1
                    best_dist_j = float('inf')
                    
                    if c in r_j:
                        continue
                        
                    for pos_j in range(len(r_j) + 1):
                        new_r_j = r_j[:pos_j] + [c] + r_j[pos_j:]
                        d_j_new = _route_distance(new_r_j, dist)
                        if d_j_new < best_dist_j:
                            feasible, d, t = _check_route(new_r_j, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                            if feasible:
                                best_dist_j = d
                                best_pos_j = pos_j
                                break
                    
                    if best_pos_j != -1:
                        new_r_j = r_j[:best_pos_j] + [c] + r_j[best_pos_j:]
                        new_r_i = r_i[:pos_i] + r_i[pos_i+1:]
                        
                        d_i_new = _route_distance(new_r_i, dist) if new_r_i else 0.0
                        d_j_new = best_dist_j
                        
                        total_new = d_i_new + d_j_new
                        total_old = route_dists[i] + route_dists[j]
                        
                        if total_new < total_old - 1e-9:
                            if new_r_i:
                                feasible_i, _, _ = _check_route(new_r_i, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                            else:
                                feasible_i = True
                            
                            if feasible_i:
                                routes[i] = new_r_i
                                routes[j] = new_r_j
                                route_dists[i] = d_i_new
                                route_dists[j] = d_j_new
                                improved = True
                                break
                    if improved:
                        break
            if improved:
                break
            
            if not r_i:
                routes.pop(i)
                route_dists.pop(i)
                break
        
        if improved:
            continue
            
    return routes


def _solve_instance(instance, time_limit=45):
    """Solve a single instance."""
    start_time = time.time()
    deadline = start_time + time_limit
    
    coords = instance["coords"]
    dist = _dist_matrix(coords)
    
    battery_max = instance["battery"]
    capacity = instance["capacity"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    n_customers = instance["n_customers"]
    
    construct_time = min(3, time_limit * 0.1)
    routes = _construct_savings_routes(instance, dist, construct_time)
    
    all_served = set()
    for r in routes:
        all_served.update(r)
    missing = [c for c in range(1, n_customers + 1) if c not in all_served]
    for c in missing:
        routes.append([c])
    
    ls_time = deadline - time.time() - 2
    if ls_time > 1:
        routes = _local_search(routes, dist, instance, ls_time)
    
    valid_routes = []
    for r in routes:
        feasible, _, _ = _check_route(r, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
        if feasible:
            valid_routes.append(r)
            
    # Deduplicate: Ensure each customer appears at most once
    seen = set()
    cleaned_routes = []
    for r in valid_routes:
        new_r = []
        for c in r:
            if c not in seen:
                new_r.append(c)
                seen.add(c)
            else:
                # Customer already served, skip this occurrence
                pass
        if new_r:
            # Re-check feasibility after removal (removal makes it easier, so it should still be feasible)
            feasible, _, _ = _check_route(new_r, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
            if feasible:
                cleaned_routes.append(new_r)
    
    valid_routes = cleaned_routes
    
    # Check for missing customers
    all_served = set()
    for r in valid_routes:
        all_served.update(r)
    
    missing = [c for c in range(1, n_customers + 1) if c not in all_served]
    for c in missing:
        # Try to insert into existing routes first
        inserted = False
        for i in range(len(valid_routes)):
            r = valid_routes[i]
            for pos in range(len(r) + 1):
                new_r = r[:pos] + [c] + r[pos:]
                feasible, _, _ = _check_route(new_r, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                if feasible:
                    valid_routes[i] = new_r
                    inserted = True
                    break
            if inserted:
                break
        if not inserted:
            # Create new route
            new_r = [c]
            feasible, _, _ = _check_route(new_r, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
            if feasible:
                valid_routes.append(new_r)
            else:
                # If even a singleton is infeasible (e.g. battery/time), we have a problem.
                # But usually singletons are feasible.
                valid_routes.append(new_r)
    
    return valid_routes


def plan(instance):
    """Main entry point."""
    start_time = time.time()
    per_instance_time = 45
    routes = _solve_instance(instance, per_instance_time)
    return routes
# EVOLVE-BLOCK-END
