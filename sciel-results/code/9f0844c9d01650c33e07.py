"""Memetic Algorithm for EV Routing with Time Windows and Battery Constraints."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random

DEADLINE_SECONDS = 115.0

def _validate_route(route, D, ready, due, service, demands, is_station, 
                    battery_max, recharge_rate, speed, horizon, capacity):
    """
    Validate a single route. Returns (is_valid, total_distance, total_time).
    """
    if not route:
        return True, 0.0, 0.0
        
    t = 0.0
    b = battery_max
    d_sum = 0.0
    total_dist = 0.0
    
    prev = 0
    for node in route:
        dist = D[prev, node]
        total_dist += dist
        
        # Check battery
        b -= dist
        if b < -1e-9:
            return False, total_dist, t
            
        # Check time
        t += dist / speed
        
        if is_station[node]:
            # Recharge
            t += (battery_max - b) * recharge_rate
            b = battery_max
        else:
            # Customer
            d_sum += demands[node]
            if d_sum > capacity:
                return False, total_dist, t
                
            # Time window
            start_service = max(t, ready[node])
            if start_service > due[node] + 1e-9:
                return False, total_dist, t
            t = start_service + service[node]
            
        prev = node
        
    # Return to depot
    dist_back = D[prev, 0]
    total_dist += dist_back
    b -= dist_back
    if b < -1e-9:
        return False, total_dist, t
    t += dist_back / speed
    
    if t > horizon + 1e-9:
        return False, total_dist, t
        
    return True, total_dist, t

def _validate_solution(routes, D, ready, due, service, demands, is_station, 
                       battery_max, recharge_rate, speed, horizon, capacity, 
                       customers):
    """
    Validate the entire solution.
    Returns (is_valid, total_distance).
    """
    total_dist = 0.0
    visited = set()
    
    for route in routes:
        valid, dist, _ = _validate_route(route, D, ready, due, service, demands, 
                                         is_station, battery_max, recharge_rate, 
                                         speed, horizon, capacity)
        if not valid:
            return False, float('inf')
        total_dist += dist
        
        for node in route:
            if not is_station[node]:
                if node in visited:
                    return False, float('inf')
                visited.add(node)
                
    if visited != set(customers):
        return False, float('inf')
        
    return True, total_dist

def _construct_nearest_neighbor(instance, D, ready, due, service, demands, is_station,
                                battery_max, recharge_rate, speed, horizon, capacity):
    """
    Construct a feasible solution using Nearest Neighbor heuristic.
    """
    customers = list(range(1, instance["n_customers"] + 1))
    unvisited = set(customers)
    routes = []
    
    while unvisited:
        # Start a new route
        route = []
        prev = 0
        t = 0.0
        b = battery_max
        d_sum = 0.0
        feasible = True
        
        # Pick a starting customer (random or closest to depot)
        # For diversity, we can pick randomly from unvisited
        start = random.choice(list(unvisited))
        unvisited.remove(start)
        
        # Check if start is feasible
        dist_start = D[0, start]
        b_check = battery_max - dist_start
        t_check = dist_start / speed
        if b_check >= -1e-9:
            start_service = max(t_check, ready[start])
            if start_service <= due[start] + 1e-9:
                t = start_service + service[start]
                b = b_check
                d_sum = demands[start]
                route.append(start)
                prev = start
                
                # Try to add more customers
                while unvisited:
                    best_cust = -1
                    best_dist = float('inf')
                    
                    for c in unvisited:
                        dist = D[prev, c]
                        # Heuristic: prefer closer
                        if dist < best_dist:
                            # Quick check
                            b_new = b - dist
                            if b_new < -1e-9:
                                continue
                            t_new = t + dist / speed
                            start_s = max(t_new, ready[c])
                            if start_s > due[c] + 1e-9:
                                continue
                            d_new = d_sum + demands[c]
                            if d_new > capacity:
                                continue
                            
                            best_dist = dist
                            best_cust = c
                            
                    if best_cust == -1:
                        break
                        
                    # Add best_cust
                    dist = D[prev, best_cust]
                    b -= dist
                    t += dist / speed
                    start_s = max(t, ready[best_cust])
                    t = start_s + service[best_cust]
                    d_sum += demands[best_cust]
                    route.append(best_cust)
                    prev = best_cust
                    unvisited.remove(best_cust)
                    
        # Check if route is feasible (including return)
        if route:
            valid, dist, t_final = _validate_route(route, D, ready, due, service, demands, 
                                                   is_station, battery_max, recharge_rate, 
                                                   speed, horizon, capacity)
            if valid:
                routes.append(route)
            else:
                # If invalid, we might need to fix it or discard and retry
                # For simplicity in construction, we keep it if it was built greedily 
                # but the final validation will catch issues. 
                # However, to ensure feasibility, we should probably split or discard.
                # Let's just keep it for now, the local search will fix or we discard in selection.
                routes.append(route)
                
    return routes

def _crossover(route1, route2):
    """
    Simple crossover: take random segments from each parent.
    Returns a new route (list of nodes).
    """
    if not route1 or not route2:
        return []
        
    n1 = len(route1)
    n2 = len(route2)
    
    # Pick random cut points
    cut1 = random.randint(0, n1)
    cut2 = random.randint(0, n2)
    
    child = route1[:cut1] + route2[cut2:]
    return child

def _mutate(route, D, ready, due, service, demands, is_station, 
            battery_max, recharge_rate, speed, horizon, capacity):
    """
    Apply random mutations: 2-opt, Or-opt (move 1 node).
    Returns mutated route.
    """
    if len(route) < 2:
        return route
        
    new_route = route[:]
    mutation_type = random.choice(['2opt', 'move'])
    
    if mutation_type == '2opt':
        i = random.randint(0, len(new_route) - 2)
        j = random.randint(i + 1, len(new_route) - 1)
        new_route[i:j+1] = new_route[i:j+1][::-1]
    elif mutation_type == 'move':
        i = random.randint(0, len(new_route) - 1)
        j = random.randint(0, len(new_route) - 1)
        if i != j:
            node = new_route.pop(i)
            new_route.insert(j, node)
            
    return new_route

def _local_search(routes, D, ready, due, service, demands, is_station, 
                  battery_max, recharge_rate, speed, horizon, capacity, 
                  time_limit):
    """
    Apply intra-route 2-opt and inter-route reinsertion.
    """
    improved = True
    start_ls = time.time()
    
    while improved:
        improved = False
        if time.time() - start_ls > time_limit:
            break
            
        # Intra-route 2-opt
        for r_idx in range(len(routes)):
            route = routes[r_idx]
            n = len(route)
            if n < 2:
                continue
                
            best_saving = 0
            best_i, best_j = -1, -1
            
            for i in range(n - 1):
                for j in range(i + 1, n):
                    prev_i = 0 if i == 0 else route[i-1]
                    next_j = 0 if j == n-1 else route[j+1]
                    
                    old_dist = D[prev_i, route[i]] + D[route[j], next_j]
                    new_dist = D[prev_i, route[j]] + D[route[i], next_j]
                    
                    saving = old_dist - new_dist
                    if saving > best_saving:
                        # Check feasibility
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        valid, _, _ = _validate_route(new_route, D, ready, due, service, demands, 
                                                      is_station, battery_max, recharge_rate, 
                                                      speed, horizon, capacity)
                        if valid:
                            best_saving = saving
                            best_i, best_j = i, j
                            
            if best_i != -1:
                routes[r_idx] = route[:best_i] + route[best_i:best_j+1][::-1] + route[best_j+1:]
                improved = True
                
        # Inter-route reinsertion (simple version: move one node to another route)
        # This is expensive, so we do it less frequently or with a limit
        if time.time() - start_ls > time_limit * 0.5:
            continue
            
        for r_idx in range(len(routes)):
            route = routes[r_idx]
            if len(route) <= 1:
                continue
                
            # Try moving each node to another route
            for i in range(len(route)):
                node = route[i]
                # Try inserting into other routes
                for o_idx in range(len(routes)):
                    if o_idx == r_idx:
                        continue
                    other_route = routes[o_idx]
                    for pos in range(len(other_route) + 1):
                        # Construct candidate
                        new_route_r = route[:i] + route[i+1:]
                        new_route_o = other_route[:pos] + [node] + other_route[pos:]
                        
                        # Check feasibility of both
                        valid_r, _, _ = _validate_route(new_route_r, D, ready, due, service, demands, 
                                                        is_station, battery_max, recharge_rate, 
                                                        speed, horizon, capacity)
                        valid_o, _, _ = _validate_route(new_route_o, D, ready, due, service, demands, 
                                                        is_station, battery_max, recharge_rate, 
                                                        speed, horizon, capacity)
                        
                        if valid_r and valid_o:
                            # Check if distance improves
                            old_dist_r = sum(D[route[k], route[k+1]] for k in range(len(route)-1)) + D[0, route[0]] + D[route[-1], 0]
                            old_dist_o = sum(D[other_route[k], other_route[k+1]] for k in range(len(other_route)-1)) + D[0, other_route[0]] + D[other_route[-1], 0]
                            
                            new_dist_r = sum(D[new_route_r[k], new_route_r[k+1]] for k in range(len(new_route_r)-1)) + D[0, new_route_r[0]] + D[new_route_r[-1], 0] if new_route_r else 0
                            new_dist_o = sum(D[new_route_o[k], new_route_o[k+1]] for k in range(len(new_route_o)-1)) + D[0, new_route_o[0]] + D[new_route_o[-1], 0] if new_route_o else 0
                            
                            if (new_dist_r + new_dist_o) < (old_dist_r + old_dist_o) - 1e-9:
                                routes[r_idx] = new_route_r
                                routes[o_idx] = new_route_o
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
                    
    return routes

def plan(instance):
    start_time = time.time()
    
    # 1. Precompute Data
    coords = np.array(instance["coords"], dtype=np.float64)
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])
    n_customers = instance["n_customers"]
    n_nodes = len(coords)
    
    is_station = np.zeros(n_nodes, dtype=bool)
    for s in stations:
        is_station[s] = True
    
    # Distance Matrix
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    D = np.sqrt(np.sum(diff**2, axis=-1))
    
    customers = list(range(1, n_customers + 1))
    
    # 2. Initialize Population
    pop_size = 20
    population = []
    for _ in range(pop_size):
        routes = _construct_nearest_neighbor(instance, D, ready, due, service, demands, is_station,
                                             battery_max, recharge_rate, speed, horizon, capacity)
        population.append(routes)
        
    # 3. Evolutionary Loop
    best_routes = None
    best_dist = float('inf')
    
    # Evaluate initial population
    for routes in population:
        valid, dist = _validate_solution(routes, D, ready, due, service, demands, is_station, 
                                         battery_max, recharge_rate, speed, horizon, capacity, customers)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = [r[:] for r in routes]
            
    generation = 0
    while time.time() - start_time < DEADLINE_SECONDS:
        generation += 1
        new_population = []
        
        # Elitism: keep best
        new_population.append([r[:] for r in best_routes])
        
        # Selection and Crossover
        for _ in range(pop_size - 1):
            # Tournament selection
            r1 = random.choice(population)
            r2 = random.choice(population)
            
            # Simple crossover: mix routes
            # For simplicity, we just pick a random route from one parent and mutate
            # Or we can do a proper route-level crossover
            child_routes = []
            # Shuffle customers from parents
            all_customers = []
            for r in r1:
                for c in r:
                    if not is_station[c]:
                        all_customers.append(c)
                        
            random.shuffle(all_customers)
            
            # Reconstruct using nearest neighbor with shuffled start? 
            # Or just take r1 and mutate
            child_routes = [r[:] for r in r1]
            
            # Apply local search to child
            # To save time, only do local search on some children
            if random.random() < 0.5:
                time_remaining = DEADLINE_SECONDS - (time.time() - start_time)
                if time_remaining > 1:
                    ls_time = min(1.0, time_remaining * 0.1)
                    child_routes = _local_search(child_routes, D, ready, due, service, demands, is_station, 
                                                 battery_max, recharge_rate, speed, horizon, capacity, ls_time)
            
            new_population.append(child_routes)
            
        # Evaluate new population
        for routes in new_population:
            valid, dist = _validate_solution(routes, D, ready, due, service, demands, is_station, 
                                             battery_max, recharge_rate, speed, horizon, capacity, customers)
            if valid and dist < best_dist:
                best_dist = dist
                best_routes = [r[:] for r in routes]
                
        population = new_population
        
        # Check time
        if time.time() - start_time > DEADLINE_SECONDS * 0.9:
            break
            
    # 4. Final Local Search on Best
    if best_routes and time.time() - start_time < DEADLINE_SECONDS * 0.95:
        time_remaining = DEADLINE_SECONDS - (time.time() - start_time) - 1.0
        if time_remaining > 0:
            best_routes = _local_search(best_routes, D, ready, due, service, demands, is_station, 
                                        battery_max, recharge_rate, speed, horizon, capacity, time_remaining)
            
    # 5. Final Validation
    if best_routes:
        valid, dist = _validate_solution(best_routes, D, ready, due, service, demands, is_station, 
                                         battery_max, recharge_rate, speed, horizon, capacity, customers)
        if valid:
            return best_routes
            
    # Fallback: return a valid single-route-per-customer solution if possible
    fallback_routes = []
    for c in customers:
        fallback_routes.append([c])
    return fallback_routes

# EVOLVE-BLOCK-END
