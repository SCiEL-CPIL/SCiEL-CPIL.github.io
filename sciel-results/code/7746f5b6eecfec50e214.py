"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


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


def _solve_with_savings(instance, dist, time_limit=20):
    """Solve using Clarke-Wright savings heuristic."""
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
    
    # Start with each customer as its own route
    routes = [[i] for i in range(1, n_customers + 1)]
    
    # Calculate savings
    savings = []
    for i in range(n_customers):
        for j in range(i + 1, n_customers):
            ci = i + 1
            cj = j + 1
            s = dist[0][ci] + dist[0][cj] - dist[ci][cj]
            savings.append((s, ci, cj))
    
    # Sort by savings descending
    savings.sort(reverse=True)
    
    # Track which route each customer belongs to
    route_of = {}
    for idx, r in enumerate(routes):
        for c in r:
            route_of[c] = idx
    
    # Try to merge routes
    for s, ci, cj in savings:
        if time.time() > deadline - 2:
            break
        
        ri = route_of.get(ci, -1)
        rj = route_of.get(cj, -1)
        if ri == -1 or rj == -1 or ri == rj:
            continue
        
        # Try both orderings
        best_new_route = None
        best_dist = float('inf')
        
        for order in [[routes[ri], routes[rj]], [routes[rj], routes[ri]]]:
            merged = order[0] + order[1]
            feasible, total_dist, _ = _check_route(merged, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
            if feasible and total_dist < best_dist:
                best_dist = total_dist
                best_new_route = merged
        
        if best_new_route is not None:
            routes[ri] = best_new_route
            routes[rj] = []
            for c in best_new_route:
                route_of[c] = ri
    
    # Remove empty routes
    routes = [r for r in routes if r]
    
    return routes


def _two_opt_improve(route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon, deadline):
    """Apply 2-opt improvements to a route."""
    if len(route) < 2:
        return route
    
    n = len(route)
    improved = True
    max_passes = 5
    
    while improved and max_passes > 0:
        improved = False
        max_passes -= 1
        for i in range(n - 1):
            if time.time() > deadline:
                return route
            for j in range(i + 1, n):
                if time.time() > deadline:
                    return route
                prev_i = route[i-1] if i > 0 else 0
                next_j = route[j+1] if j < n - 1 else 0
                
                d_old = dist[prev_i][route[i]] + dist[route[j]][next_j]
                d_new = dist[prev_i][route[j]] + dist[route[i]][next_j]
                
                if d_new < d_old - 1e-9:
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    feasible, _, _ = _check_route(new_route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                    if feasible:
                        route = new_route
                        improved = True
                        break
            if improved:
                break
    
    return route

def _or_opt_improve(route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon, deadline):
    """Apply Or-Opt improvements (move 1, 2, or 3 consecutive nodes)."""
    n = len(route)
    if n < 2:
        return route
    
    improved = True
    max_passes = 3
    while improved and max_passes > 0:
        improved = False
        max_passes -= 1
        
        for L in [1, 2, 3]:
            if n < L + 1: continue
            for i in range(n - L + 1):
                if time.time() > deadline:
                    return route
                # Nodes to move
                moving = route[i:i+L]
                # Remaining route
                rest = route[:i] + route[i+L:]
                
                # Try inserting at every position in rest
                for j in range(len(rest) + 1):
                    if j == i or j == i+L:
                        continue
                    new_route = rest[:j] + moving + rest[j:]
                    # Check if it's the same as original
                    if new_route == route:
                        continue
                    
                    feasible, _, _ = _check_route(new_route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                    if feasible:
                        # Calculate distance change to accept only if better
                        # Simple distance calc for new route
                        d_new = 0
                        prev = 0
                        for v in new_route:
                            d_new += dist[prev][v]
                            prev = v
                        d_new += dist[prev][0]
                        
                        d_old = 0
                        prev = 0
                        for v in route:
                            d_old += dist[prev][v]
                            prev = v
                        d_old += dist[prev][0]
                        
                        if d_new < d_old - 1e-9:
                            route = new_route
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    return route

def _local_search(routes, dist, instance, battery_max, capacity, speed, recharge_rate, horizon, deadline):
    """Inter-route and intra-route local search."""
    n_customers = instance["n_customers"]
    
    while time.time() < deadline:
        improvement_found = False
        
        # 1. Inter-route Relocate
        for i in range(len(routes)):
            for j in range(len(routes)):
                if i == j or not routes[i] or not routes[j]:
                    continue
                # Move customer from i to j
                for k in range(len(routes[i])):
                    c = routes[i][k]
                    if c > n_customers: continue # Only customers can be moved? No, stations are fixed? 
                    # In this problem, stations are not in the route list returned? 
                    # "Return a list of routes, each a list of node ids; the depot is implicit... stations may repeat"
                    # The route list contains node ids. If we move a station, it might break logic.
                    # But stations are optional visits. Let's only move customers (1..n_customers).
                    
                    # Remove c from i
                    route_i_new = routes[i][:k] + routes[i][k+1:]
                    # Insert c into j at best position
                    best_pos = -1
                    best_dist = float('inf')
                    
                    # Try all positions in j
                    for pos in range(len(routes[j]) + 1):
                        route_j_new = routes[j][:pos] + [c] + routes[j][pos:]
                        # Check feasibility of both
                        feas_i, dist_i, _ = _check_route(route_i_new, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                        feas_j, dist_j, _ = _check_route(route_j_new, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                        
                        if feas_i and feas_j:
                            total_d = dist_i + dist_j
                            if total_d < best_dist:
                                best_dist = total_d
                                best_pos = pos
                                best_route_i = route_i_new
                                best_route_j = route_j_new
                    
                    if best_pos != -1:
                        # Calculate old total dist
                        feas_i_old, dist_i_old, _ = _check_route(routes[i], dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                        feas_j_old, dist_j_old, _ = _check_route(routes[j], dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                        old_total = dist_i_old + dist_j_old
                        
                        if best_dist < old_total - 1e-9:
                            routes[i] = best_route_i
                            routes[j] = best_route_j
                            improvement_found = True
                            break
                if improvement_found:
                    break
            if improvement_found:
                break
        
        if improvement_found:
            continue
            
        # 2. Inter-route Swap
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                if not routes[i] or not routes[j]:
                    continue
                # Swap c from i and d from j
                for k in range(len(routes[i])):
                    c = routes[i][k]
                    if c > n_customers: continue
                    for l in range(len(routes[j])):
                        d = routes[j][l]
                        if d > n_customers: continue
                        
                        route_i_new = routes[i][:]
                        route_i_new[k] = d
                        route_j_new = routes[j][:]
                        route_j_new[l] = c
                        
                        feas_i, dist_i, _ = _check_route(route_i_new, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                        feas_j, dist_j, _ = _check_route(route_j_new, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                        
                        if feas_i and feas_j:
                            old_total = 0
                            for r in [routes[i], routes[j]]:
                                _, d_old, _ = _check_route(r, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                                old_total += d_old
                            
                            if dist_i + dist_j < old_total - 1e-9:
                                routes[i] = route_i_new
                                routes[j] = route_j_new
                                improvement_found = True
                                break
                    if improvement_found:
                        break
                if improvement_found:
                    break
            if improvement_found:
                break
        
        if improvement_found:
            continue
            
        # 3. Intra-route 2-Opt and Or-Opt
        for i in range(len(routes)):
            if time.time() > deadline:
                break
            old_route = routes[i][:]
            routes[i] = _two_opt_improve(routes[i], dist, instance, battery_max, capacity, speed, recharge_rate, horizon, deadline)
            routes[i] = _or_opt_improve(routes[i], dist, instance, battery_max, capacity, speed, recharge_rate, horizon, deadline)
            
            if routes[i] != old_route:
                improvement_found = True
                break
        
        if not improvement_found:
            break
            
    return routes

def _solve_instance(instance, time_limit=25):
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
    
    # Use savings heuristic
    routes = _solve_with_savings(instance, dist, time_limit - 5)
    
    # Apply local search
    routes = _local_search(routes, dist, instance, battery_max, capacity, speed, recharge_rate, horizon, deadline - 2)
    
    # Final verification
    valid_routes = []
    for r in routes:
        feasible, _, _ = _check_route(r, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
        if feasible:
            valid_routes.append(r)
    
    # Check all customers are served
    all_served = set()
    for r in valid_routes:
        all_served.update(r)
    
    n_customers = instance["n_customers"]
    missing = [c for c in range(1, n_customers + 1) if c not in all_served]
    
    for c in missing:
        valid_routes.append([c])
    
    return valid_routes


def plan(instance):
    """Main entry point."""
    start_time = time.time()
    
    # 3 instances, 150s total, leave 10s buffer -> 40s per instance
    per_instance_time = 40
    
    routes = _solve_instance(instance, per_instance_time)
    
    return routes
# EVOLVE-BLOCK-END
