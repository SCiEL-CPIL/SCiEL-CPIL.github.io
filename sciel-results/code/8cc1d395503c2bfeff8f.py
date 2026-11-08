# EVOLVE-BLOCK-START
import numpy as np
import time
from math import sqrt

def _dist(coords, i, j):
    dx = coords[i, 0] - coords[j, 0]
    dy = coords[i, 1] - coords[j, 1]
    return sqrt(dx * dx + dy * dy)

def _simulate_route(route_customers, dist, ready, due, service, battery, speed, 
                    recharge_rate, horizon, stations, n_nodes, coords):
    """Simulate a route with charging stops. Returns (feasible, total_dist, total_time)."""
    if not route_customers:
        return True, 0.0, 0.0
    
    total_dist = 0.0
    total_time = 0.0
    current_node = 0
    bat = battery
    
    for c in route_customers:
        d = dist[current_node, c]
        travel_time = d / speed
        
        # Check if we can reach this customer with current battery
        if bat < d:
            # Need to charge - find best station to visit first
            # Try each station: go to station, charge, then go to customer
            best_option = None
            for s in stations:
                d_to_station = dist[current_node, s]
                if bat >= d_to_station:
                    d_station_to_cust = dist[s, c]
                    charge_time = (bat - d_to_station) * recharge_rate
                    # After charging at station, battery is full
                    if d_station_to_cust <= battery:
                        total_time_to_cust = total_time + d_to_station / speed + charge_time + d_station_to_cust / speed
                        total_dist_so_far = total_dist + d_to_station + d_station_to_cust
                        arr_time = total_time_to_cust
                        start_service = max(arr_time, ready[c])
                        if start_service <= due[c] and total_dist_so_far + dist[c, 0] < 10000:
                            if best_option is None or total_dist_so_far < best_option[0]:
                                best_option = (total_dist_so_far, s, total_time_to_cust, start_service)
            
            if best_option is None:
                return False, total_dist, total_time
            
            total_dist, station_used, total_time, start_service = best_option
            total_time = start_service + service[c]
            bat = battery - dist[station_used, c]
            current_node = c
        else:
            total_dist += d
            total_time += travel_time
            bat -= d
            arr_time = total_time
            start_service = max(arr_time, ready[c])
            if start_service > due[c]:
                return False, total_dist, total_time
            total_time = start_service + service[c]
            current_node = c
    
    # Return to depot
    d = dist[current_node, 0]
    if bat < d:
        # Need to charge before returning
        best_option = None
        for s in stations:
            d_to_station = dist[current_node, s]
            if bat >= d_to_station:
                d_station_to_depot = dist[s, 0]
                charge_time = (bat - d_to_station) * recharge_rate
                if d_station_to_depot <= battery:
                    total_time_return = total_time + d_to_station / speed + charge_time + d_station_to_depot / speed
                    total_dist_return = total_dist + d_to_station + d_station_to_depot
                    if total_time_return <= horizon:
                        if best_option is None or total_dist_return < best_option[0]:
                            best_option = (total_dist_return, total_time_return)
        
        if best_option is None:
            return False, total_dist, total_time
        total_dist, total_time = best_option
    else:
        total_dist += d
        total_time += d / speed
        bat -= d
        if total_time > horizon:
            return False, total_dist, total_time
    
    return True, total_dist, total_time

def _solve_instance(instance, time_limit=40):
    coords = np.array(instance["coords"])
    n_customers = instance["n_customers"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    # Precompute distances
    n_nodes = len(coords)
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = sqrt((coords[i, 0] - coords[j, 0]) ** 2 + (coords[i, 1] - coords[j, 1]) ** 2)
            dist[i, j] = d
            dist[j, i] = d
    
    customers = list(range(1, n_customers + 1))
    
    # Nearest-neighbor insertion with full simulation
    # Start with a route from depot, greedily add nearest feasible customer
    routes = []
    unserved = set(customers)
    
    while unserved:
        start_time = time.time()
        
        # Start new route from depot
        current_route = []
        current_demand = 0
        current_node = 0
        bat = battery
        total_time = 0.0
        total_dist = 0.0
        
        while unserved:
            # Find the nearest unserved customer that can be added feasibly
            best_customer = None
            best_cost = float('inf')
            
            for c in unserved:
                if current_demand + demands[c] > capacity:
                    continue
                
                d = dist[current_node, c]
                
                # Check battery
                if bat < d:
                    # Try charging at a station first
                    found_feasible = False
                    for s in stations:
                        d_to_s = dist[current_node, s]
                        if bat >= d_to_s:
                            d_s_to_c = dist[s, c]
                            if d_s_to_c <= battery:
                                charge_time = (bat - d_to_s) * recharge_rate
                                arr_time = total_time + d_to_s / speed + charge_time + d_s_to_c / speed
                                start_service = max(arr_time, ready[c])
                                if start_service <= due[c]:
                                    # Estimate total route distance
                                    est_dist = total_dist + d_to_s + d_s_to_c + dist[c, 0]
                                    if est_dist < best_cost:
                                        best_cost = est_dist
                                        best_customer = c
                                        found_feasible = True
                                        break
                    if found_feasible:
                        continue
                
                # Direct reach
                travel_time = d / speed
                arr_time = total_time + travel_time
                start_service = max(arr_time, ready[c])
                
                if start_service <= due[c]:
                    # Estimate total route distance
                    est_dist = total_dist + d + dist[c, 0]
                    if est_dist < best_cost:
                        best_cost = est_dist
                        best_customer = c
            
            if best_customer is None:
                break
            
            # Add the customer to the route
            c = best_customer
            unserved.discard(c)
            current_route.append(c)
            current_demand += demands[c]
            
            # Update state - need to re-simulate to get accurate time/battery
            # For simplicity, approximate: assume direct travel (will be verified later)
            d = dist[current_node, c]
            total_dist += d
            total_time += d / speed
            bat -= d
            start_service = max(total_time, ready[c])
            total_time = start_service + service[c]
            current_node = c
        
        if current_route:
            routes.append(current_route)
        
        # Time check
        if time.time() - start_time > time_limit / max(1, len(routes) + 1):
            # For remaining customers, assign one per route
            for c in sorted(unserved, key=lambda x: due[x]):
                routes.append([c])
            break
    
    if unserved:
        for c in sorted(unserved, key=lambda x: due[x]):
            routes.append([c])
    
    # Optimize each route with 2-opt + feasibility check
    def route_dist(route_customers):
        if not route_customers:
            return 0.0
        total = 0.0
        prev = 0
        for c in route_customers:
            total += dist[prev, c]
            prev = c
        total += dist[prev, 0]
        return total
    
    def two_opt_with_feasibility(route_customers, max_iters=30):
        if len(route_customers) < 2:
            return route_customers
        
        best = list(route_customers)
        best_dist = route_dist(best)
        # Verify initial feasibility
        feasible, _, _ = _simulate_route(best, dist, ready, due, service, battery, speed, 
                                         recharge_rate, horizon, stations, n_nodes, coords)
        if not feasible:
            return route_customers
        
        improved = True
        iterations = 0
        while improved and iterations < max_iters:
            improved = False
            iterations += 1
            for i in range(len(best) - 1):
                for j in range(i + 1, len(best)):
                    new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                    new_dist = route_dist(new_route)
                    if new_dist < best_dist - 1e-9:
                        feasible, _, _ = _simulate_route(new_route, dist, ready, due, service, 
                                                         battery, speed, recharge_rate, horizon, 
                                                         stations, n_nodes, coords)
                        if feasible:
                            best = new_route
                            best_dist = new_dist
                            improved = True
        return best
    
    optimized_routes = []
    for route in routes:
        opt_route = two_opt_with_feasibility(route, max_iters=20)
        # Final feasibility check
        feasible, _, _ = _simulate_route(opt_route, dist, ready, due, service, battery, speed, 
                                         recharge_rate, horizon, stations, n_nodes, coords)
        if feasible:
            optimized_routes.append(opt_route)
        else:
            # Fall back to original or split
            feasible_orig, _, _ = _simulate_route(route, dist, ready, due, service, battery, speed, 
                                                  recharge_rate, horizon, stations, n_nodes, coords)
            if feasible_orig:
                optimized_routes.append(route)
            else:
                for c in route:
                    optimized_routes.append([c])
    
    return optimized_routes

def plan(instance):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    routes = _solve_instance(instance, time_limit=30)
    
    # Final time check - if we're running out of time, return what we have
    if time.time() > deadline - 5:
        pass  # Already returned
    
    return routes
# EVOLVE-BLOCK-END
