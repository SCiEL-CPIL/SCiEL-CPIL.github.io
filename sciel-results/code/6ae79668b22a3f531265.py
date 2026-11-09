"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from itertools import combinations


def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    coords = np.array(instance["coords"])
    demands = np.array(instance["demands"])
    ready = np.array(instance["ready"])
    due = np.array(instance["due"])
    service = np.array(instance["service"])
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
        for j in range(i + 1, n_nodes):
            d = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
            dist[i, j] = d
            dist[j, i] = d

    customers = list(range(1, n_customers + 1))

    def check_route(route):
        """Check if a route is feasible. Returns (feasible, total_distance)."""
        if not route:
            return True, 0.0
        
        # Check capacity
        total_demand = sum(demands[v] for v in route)
        if total_demand > capacity:
            return False, 0.0

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
                return False, 0.0
            
            # Check if we need to charge before service
            # We can charge at stations only, but here we're just checking customer visit
            # Actually, we need to handle charging at stations within the route
            # For simplicity in this check, assume no stations in route (stations are separate nodes)
            
            # At customer
            if t < ready[v]:
                t = ready[v]
            if t > due[v]:
                return False, 0.0
            t += service[v]
            prev = v
        
        # Return to depot
        d = dist[prev, 0]
        bat -= d
        t += d / speed
        total_dist += d
        
        if bat < 0:
            return False, 0.0
        if t > horizon:
            return False, 0.0
        
        return True, total_dist

    def greedy_construct(customer_list, stations_list):
        """Construct routes using greedy nearest neighbor with time windows."""
        remaining = set(customer_list)
        routes = []
        
        while remaining:
            # Start new route from depot
            route = []
            prev = 0
            t = 0.0
            bat = battery
            total_dist = 0.0
            load = 0
            
            while remaining:
                # Find best next customer
                best_customer = None
                best_score = float('inf')
                
                for v in remaining:
                    d = dist[prev, v]
                    bat_after = bat - d
                    t_after = t + d / speed
                    
                    if bat_after < 0:
                        continue
                    if load + demands[v] > capacity:
                        continue
                    
                    # Check time window
                    if t_after < ready[v]:
                        t_after = ready[v]
                    if t_after > due[v]:
                        continue
                    
                    # Score: distance + time penalty
                    score = d + (t_after - ready[v]) * 0.1
                    
                    if score < best_score:
                        best_score = score
                        best_customer = v
                
                if best_customer is None:
                    break
                
                # Add customer
                v = best_customer
                d = dist[prev, v]
                bat -= d
                t += d / speed
                total_dist += d
                load += demands[v]
                
                if t < ready[v]:
                    t = ready[v]
                t += service[v]
                
                route.append(v)
                remaining.discard(v)
                prev = v
            
            if route:
                # Check if we can return to depot
                d = dist[prev, 0]
                if bat - d >= 0 and t + d / speed <= horizon:
                    routes.append(route)
                    total_dist += d
                else:
                    # Can't complete this route, add to routes anyway (will be invalid)
                    routes.append(route)
        
        return routes

    # Initial solution: one customer per route (always feasible)
    best_routes = [[v] for v in customers]
    best_distance = sum(dist[0, v] * 2 for v in customers)

    # Try to improve by merging customers into routes
    # Sort customers by due time for priority
    sorted_customers = sorted(customers, key=lambda v: due[v])
    
    # Try greedy construction
    if time.time() < deadline:
        greedy_routes = greedy_construct(sorted_customers, stations)
        # Validate greedy routes
        all_customers = []
        valid = True
        total_dist = 0.0
        for route in greedy_routes:
            feasible, d = check_route(route)
            if not feasible:
                valid = False
                break
            all_customers.extend(route)
            total_dist += d
        
        if valid and set(all_customers) == set(customers) and len(all_customers) == len(set(all_customers)):
            if total_dist < best_distance:
                best_routes = greedy_routes
                best_distance = total_dist

    # Try 2-opt local search on the best routes
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

    # Local search: try swapping customers between routes
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_dist = total_route_distance(best_routes)
        
        # Try moving customers from one route to another
        for i in range(len(best_routes)):
            for j in range(len(best_routes[i])):
                if time.time() > deadline:
                    break
                v = best_routes[i][j]
                # Try inserting v into other routes
                for k in range(len(best_routes)):
                    if k == i:
                        continue
                    for pos in range(len(best_routes[k]) + 1):
                        # Create new routes
                        new_routes = []
                        for r in range(len(best_routes)):
                            if r == i:
                                new_routes.append([x for x in best_routes[i] if x != v])
                            elif r == k:
                                new_route = best_routes[k][:]
                                new_route.insert(pos, v)
                                new_routes.append(new_route)
                            else:
                                new_routes.append(best_routes[r][:])
                        
                        # Check feasibility
                        feasible, d = True, 0.0
                        for route in new_routes:
                            f, dd = check_route(route)
                            if not f:
                                feasible = False
                                break
                            d += dd
                        
                        if feasible:
                            new_dist = total_route_distance(new_routes)
                            if new_dist < current_dist - 1e-6:
                                best_routes = new_routes
                                current_dist = new_dist
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    
    return best_routes
# EVOLVE-BLOCK-END
