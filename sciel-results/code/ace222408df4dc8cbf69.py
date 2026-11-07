import numpy as np
import time
import math
import random

# EVOLVE-BLOCK-START

def plan(instance):
    start_time = time.time()
    TIME_LIMIT = 110.0  # Safety margin for 120s limit
    
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    # Precompute distance matrix
    n_nodes = len(coords)
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            d = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
            dist[i][j] = d
            dist[j][i] = d

    customers = list(range(1, n_customers + 1))
    
    # 1. Savings-based construction
    # Initial routes: one customer per route
    routes = [[c] for c in customers]
    
    # Calculate savings for merging two routes
    # Savings = dist(depot, i) + dist(depot, j) - dist(i, j)
    # We want to maximize savings while maintaining feasibility
    
    def is_route_feasible(route):
        """Check if a route is feasible (capacity, battery, time windows, horizon)."""
        if not route:
            return True
        
        # Check capacity
        total_demand = sum(demands[c] for c in route)
        if total_demand > capacity:
            return False
            
        # Check battery and time
        curr_time = 0.0
        curr_battery = battery
        curr_node = 0
        
        for node in route:
            d = dist[curr_node][node]
            curr_time += d / speed
            curr_battery -= d
            
            if curr_battery < 0:
                return False
            
            # Time window
            if curr_time < ready[node]:
                curr_time = ready[node]
            if curr_time > due[node]:
                return False
            
            curr_time += service[node]
            curr_node = node
            
        # Return to depot
        d = dist[curr_node][0]
        curr_time += d / speed
        curr_battery -= d
        
        if curr_battery < 0:
            return False
        if curr_time > horizon:
            return False
            
        return True

    def calculate_savings(i, j):
        return dist[0][i] + dist[0][j] - dist[i][j]

    # Sort customers by due time for better initial construction
    customers_sorted = sorted(customers, key=lambda c: due[c])
    routes = [[c] for c in customers_sorted]
    
    # Build savings list
    savings_list = []
    for i in range(len(routes)):
        for j in range(i+1, len(routes)):
            # Only consider merging if last of i and first of j are compatible
            last_i = routes[i][-1]
            first_j = routes[j][0]
            s = calculate_savings(last_i, first_j)
            savings_list.append((s, i, j))
            
    savings_list.sort(reverse=True)
    
    # Merge routes
    route_exists = [True] * len(routes)
    route_end = [r[-1] for r in routes]
    route_start = [r[0] for r in routes]
    
    for s, i, j in savings_list:
        if s <= 0:
            break
        if not route_exists[i] or not route_exists[j]:
            continue
            
        # Check if we can merge routes i and j
        # New route: routes[i] + routes[j]
        new_route = routes[i] + routes[j]
        
        # Check capacity
        if sum(demands[c] for c in new_route) > capacity:
            continue
            
        # Check feasibility with battery
        # We need to check if battery allows the trip
        # Simple check: total distance < battery
        # More precise check would involve finding optimal charging points
        # For now, let's do a simple feasibility check
        if is_route_feasible(new_route):
            routes[i] = new_route
            route_end[i] = new_route[-1]
            route_exists[j] = False
            routes[j] = []
    
    # Filter empty routes
    routes = [r for r in routes if r]
    
    # 2. Local Search
    def route_distance(route):
        d = dist[0][route[0]]
        for i in range(len(route)-1):
            d += dist[route[i]][route[i+1]]
        d += dist[route[-1]][0]
        return d

    def try_2opt(route):
        """Try 2-opt improvement on a single route."""
        improved = True
        best_route = route[:]
        best_dist = route_distance(best_route)
        
        while improved:
            improved = False
            for i in range(len(best_route)-1):
                for j in range(i+2, len(best_route)):
                    # Reverse segment i+1 to j
                    new_route = best_route[:i+1] + best_route[i+1:j+1][::-1] + best_route[j+1:]
                    
                    if is_route_feasible(new_route):
                        new_dist = route_distance(new_route)
                        if new_dist < best_dist - 1e-6:
                            best_route = new_route
                            best_dist = new_dist
                            improved = True
                            
        return best_route

    def try_or_opt(route):
        """Try Or-opt improvement on a single route."""
        improved = True
        best_route = route[:]
        best_dist = route_distance(best_route)
        
        while improved:
            improved = False
            n = len(best_route)
            for seg_len in [1, 2, 3]:
                for i in range(n - seg_len + 1):
                    segment = best_route[i:i+seg_len]
                    rest = best_route[:i] + best_route[i+seg_len:]
                    
                    for j in range(len(rest) + 1):
                        new_route = rest[:j] + segment + rest[j:]
                        
                        if is_route_feasible(new_route):
                            new_dist = route_distance(new_route)
                            if new_dist < best_dist - 1e-6:
                                best_route = new_route
                                best_dist = new_dist
                                improved = True
                                
        return best_route

    # Apply local search to each route
    for _ in range(10):  # Multiple passes
        if time.time() - start_time > TIME_LIMIT:
            break
            
        for i in range(len(routes)):
            if time.time() - start_time > TIME_LIMIT:
                break
                
            # 2-opt
            routes[i] = try_2opt(routes[i])
            
            # Or-opt
            routes[i] = try_or_opt(routes[i])

    # 3. Inter-route moves (reassignment)
    for _ in range(5):
        if time.time() - start_time > TIME_LIMIT:
            break
            
        for i in range(len(routes)):
            if time.time() - start_time > TIME_LIMIT:
                break
                
            for j in range(len(routes[i])):
                if time.time() - start_time > TIME_LIMIT:
                    break
                    
                customer = routes[i][j]
                
                # Try moving customer to another route
                for k in range(len(routes)):
                    if k == i:
                        continue
                    
                    if time.time() - start_time > TIME_LIMIT:
                        break
                        
                    # Try inserting customer at each position in route k
                    for pos in range(len(routes[k]) + 1):
                        new_route_i = routes[i][:j] + routes[i][j+1:]
                        new_route_k = routes[k][:pos] + [customer] + routes[k][pos:]
                        
                        if not new_route_i and not new_route_k:
                            continue
                            
                        # Check capacity
                        if sum(demands[c] for c in new_route_k) > capacity:
                            continue
                            
                        # Check feasibility
                        if is_route_feasible(new_route_k) and is_route_feasible(new_route_i):
                            # Calculate distance change
                            old_dist = route_distance(routes[i]) + route_distance(routes[k])
                            new_dist = route_distance(new_route_i) + route_distance(new_route_k)
                            
                            if new_dist < old_dist - 1e-6:
                                routes[i] = new_route_i
                                routes[k] = new_route_k
                                break

    # Filter empty routes again
    routes = [r for r in routes if r]
    
    return routes

# EVOLVE-BLOCK-END
