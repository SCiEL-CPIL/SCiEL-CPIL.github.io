"""Optimized EV routing with time windows and battery constraints."""
# EVOLVE-BLOCK-START
import numpy as np
from math import sqrt
import time
import random

DEADLINE_SECONDS = 140.0  # safety margin below 150s limit


def _build_dist_matrix(coords):
    """Precompute distance matrix using vectorized operations."""
    coords = np.array(coords, dtype=np.float64)
    n_nodes = len(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff * diff, axis=2))
    return dist_matrix


def _simulate_route(route, dist_matrix, instance, n_customers, stations,
                    battery_max, speed, recharge_rate, horizon, capacity):
    """Simulate a route. Returns (valid, distance, time_at_depot, battery_at_depot)."""
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    
    # Check demand
    total_demand = sum(demands[node] for node in route if node <= n_customers)
    if total_demand > capacity:
        return False, 0.0, 0.0, 0.0
    
    # Simulate route
    battery = battery_max
    time_at = 0.0
    distance = 0.0
    current_node = 0  # depot
    
    for node in route:
        d = dist_matrix[current_node, node]
        time_at += d / speed
        battery -= d
        distance += d
        
        if battery < -1e-9:
            return False, 0.0, 0.0, 0.0
        
        if node <= n_customers:
            # Customer: check time window
            time_at = max(time_at, ready[node])
            if time_at > due[node] + 1e-9:
                return False, 0.0, 0.0, 0.0
            time_at += service[node]
        elif node in stations:
            # Charging station
            energy_needed = battery_max - battery
            time_at += energy_needed * recharge_rate
            battery = battery_max
        
        current_node = node
    
    # Return to depot
    d = dist_matrix[current_node, 0]
    time_at += d / speed
    battery -= d
    distance += d
    
    if battery < -1e-9:
        return False, 0.0, 0.0, 0.0
    if time_at > horizon + 1e-9:
        return False, 0.0, 0.0, 0.0
    
    return True, distance, time_at, battery


def _route_distance(route, dist_matrix):
    """Compute the distance of a route including depot legs."""
    total = 0.0
    current = 0
    for node in route:
        total += dist_matrix[current, node]
        current = node
    total += dist_matrix[current, 0]
    return total


def _validate_solution(routes, instance):
    """Validate all routes. Returns (valid, total_distance)."""
    n_customers = instance["n_customers"]
    coords = np.array(instance["coords"], dtype=np.float64)
    dist_matrix = _build_dist_matrix(coords)
    
    stations = set(instance["stations"])
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    capacity = instance["capacity"]
    
    served = set()
    total_distance = 0.0
    
    for route in routes:
        valid, dist, _, _ = _simulate_route(route, dist_matrix, instance, n_customers,
                                             stations, battery_max, speed, recharge_rate,
                                             horizon, capacity)
        if not valid:
            return False, 0.0
        total_distance += dist
        
        for node in route:
            if node <= n_customers:
                if node in served:
                    return False, 0.0
                served.add(node)
    
    # Check all customers served exactly once
    if served != set(range(1, n_customers + 1)):
        return False, 0.0
    
    return True, total_distance


def _greedy_construct(instance, dist_matrix, n_customers, stations, battery_max,
                      speed, recharge_rate, horizon, capacity):
    """Construct initial solution using greedy insertion."""
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    
    # Sort customers by due time (earliest deadline first)
    customers = list(range(1, n_customers + 1))
    customers.sort(key=lambda v: due[v])
    
    routes = []
    route_loads = []
    
    for cust in customers:
        best_route_idx = -1
        best_insert_pos = -1
        best_cost = float('inf')
        
        for r_idx, route in enumerate(routes):
            if route_loads[r_idx] + demands[cust] > capacity:
                continue
            
            # Try inserting at each position
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [cust] + route[pos:]
                
                valid, _, _, _ = _simulate_route(new_route, dist_matrix, instance,
                                                   n_customers, stations, battery_max,
                                                   speed, recharge_rate, horizon, capacity)
                if valid:
                    # Calculate cost increase
                    old_dist = _route_distance(route, dist_matrix)
                    new_dist = _route_distance(new_route, dist_matrix)
                    cost = new_dist - old_dist
                    if cost < best_cost:
                        best_cost = cost
                        best_route_idx = r_idx
                        best_insert_pos = pos
        
        if best_route_idx == -1:
            # Create new route
            routes.append([cust])
            route_loads.append(demands[cust])
        else:
            routes[best_route_idx].insert(best_insert_pos, cust)
            route_loads[best_route_idx] += demands[cust]
    
    return routes


def _two_opt_route(route, dist_matrix, instance, n_customers, stations, battery_max,
                   speed, recharge_rate, horizon, capacity, max_iters=100):
    """Optimize the order of nodes within a single route using 2-opt."""
    if len(route) < 2:
        return route
    
    best_route = route[:]
    best_dist = _route_distance(best_route, dist_matrix)
    
    for _ in range(max_iters):
        improved = False
        for i in range(len(best_route) - 1):
            for j in range(i + 1, len(best_route)):
                # Try reversing segment i..j
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                
                valid, _, _, _ = _simulate_route(new_route, dist_matrix, instance,
                                                   n_customers, stations, battery_max,
                                                   speed, recharge_rate, horizon, capacity)
                if valid:
                    new_dist = _route_distance(new_route, dist_matrix)
                    if new_dist < best_dist - 1e-9:
                        best_route = new_route
                        best_dist = new_dist
                        improved = True
        if not improved:
            break
    
    return best_route


def _local_search(routes, dist_matrix, instance, deadline):
    """Improve solution using intra-route 2-opt and inter-route moves."""
    n_customers = instance["n_customers"]
    demands = np.array(instance["demands"], dtype=np.float64)
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])
    
    def get_total_dist(rts):
        total = 0.0
        for route in rts:
            total += _route_distance(route, dist_matrix)
        return total
    
    def is_feasible(route):
        valid, _, _, _ = _simulate_route(route, dist_matrix, instance,
                                           n_customers, stations, battery_max,
                                           speed, recharge_rate, horizon, capacity)
        return valid
    
    def route_load(route):
        return sum(demands[node] for node in route if node <= n_customers)
    
    start_time = time.time()
    current_dist = get_total_dist(routes)
    
    # First pass: optimize each route's internal order with 2-opt
    for r_idx in range(len(routes)):
        if time.time() > deadline:
            break
        routes[r_idx] = _two_opt_route(
            routes[r_idx], dist_matrix, instance, n_customers, stations,
            battery_max, speed, recharge_rate, horizon, capacity, max_iters=50
        )
    
    current_dist = get_total_dist(routes)
    
    # Second pass: inter-route moves
    max_passes = 30
    for _pass in range(max_passes):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try moving customers between routes
        for r1 in range(len(routes)):
            if time.time() > deadline:
                break
            for pos1 in range(len(routes[r1])):
                if time.time() > deadline:
                    break
                cust = routes[r1][pos1]
                if cust > n_customers:
                    continue
                
                for r2 in range(len(routes)):
                    if time.time() > deadline:
                        break
                    if r1 == r2:
                        continue
                    if route_load(routes[r2]) + demands[cust] > capacity:
                        continue
                    
                    for pos2 in range(len(routes[r2]) + 1):
                        if time.time() > deadline:
                            break
                        
                        route1_new = routes[r1][:pos1] + routes[r1][pos1+1:]
                        route2_new = routes[r2][:pos2] + [cust] + routes[r2][pos2:]
                        
                        if is_feasible(route1_new) and is_feasible(route2_new):
                            new_routes = [r[:] for r in routes]
                            new_routes[r1] = route1_new
                            new_routes[r2] = route2_new
                            new_dist = get_total_dist(new_routes)
                            if new_dist < current_dist - 1e-9:
                                routes = new_routes
                                current_dist = new_dist
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        # Try swapping customers between routes
        if not improved and time.time() < deadline:
            for r1 in range(len(routes)):
                if time.time() > deadline:
                    break
                for pos1 in range(len(routes[r1])):
                    if time.time() > deadline:
                        break
                    cust1 = routes[r1][pos1]
                    if cust1 > n_customers:
                        continue
                    
                    for r2 in range(len(routes)):
                        if time.time() > deadline:
                            break
                        if r1 == r2:
                            continue
                        for pos2 in range(len(routes[r2])):
                            if time.time() > deadline:
                                break
                            cust2 = routes[r2][pos2]
                            if cust2 > n_customers:
                                continue
                            
                            # Check capacity after swap
                            load1 = route_load(routes[r1])
                            load2 = route_load(routes[r2])
                            if load1 - demands[cust1] + demands[cust2] > capacity:
                                continue
                            if load2 - demands[cust2] + demands[cust1] > capacity:
                                continue
                            
                            route1_new = routes[r1][:]
                            route1_new[pos1] = cust2
                            route2_new = routes[r2][:]
                            route2_new[pos2] = cust1
                            
                            if is_feasible(route1_new) and is_feasible(route2_new):
                                new_routes = [r[:] for r in routes]
                                new_routes[r1] = route1_new
                                new_routes[r2] = route2_new
                                new_dist = get_total_dist(new_routes)
                                if new_dist < current_dist - 1e-9:
                                    routes = new_routes
                                    current_dist = new_dist
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    # Remove empty routes
    routes = [r for r in routes if len(r) > 0]
    
    return routes


def plan(instance):
    """Main entry point."""
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Build distance matrix once
    coords = np.array(instance["coords"], dtype=np.float64)
    dist_matrix = _build_dist_matrix(coords)
    
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    capacity = instance["capacity"]
    
    # Get greedy solution
    routes = _greedy_construct(instance, dist_matrix, n_customers, stations,
                               battery_max, speed, recharge_rate, horizon, capacity)
    
    # Validate
    valid, dist = _validate_solution(routes, instance)
    if not valid:
        # Fallback to one-per-route
        customers = list(range(1, n_customers + 1))
        routes = [[v] for v in customers]
    
    # Local search if time permits
    if time.time() < deadline - 5:
        routes = _local_search(routes, dist_matrix, instance, deadline)
        
        # Re-validate
        valid, dist = _validate_solution(routes, instance)
        if not valid:
            # Fallback
            customers = list(range(1, n_customers + 1))
            routes = [[v] for v in customers]
    
    # Final check
    valid, dist = _validate_solution(routes, instance)
    if not valid:
        customers = list(range(1, n_customers + 1))
        routes = [[v] for v in customers]
    
    return routes

# EVOLVE-BLOCK-END
