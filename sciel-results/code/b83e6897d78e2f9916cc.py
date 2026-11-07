"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import math

def _precompute_dist_matrix(coords):
    """Precompute pairwise distance matrix."""
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((coords[i, 0] - coords[j, 0])**2 + (coords[i, 1] - coords[j, 1])**2)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    return dist_matrix

def _simulate_route(route, instance, dist_matrix):
    """
    Simulate a route and return (feasible, total_distance).
    """
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])
    
    total_demand = sum(demands[v] for v in route if v not in stations)
    if total_demand > capacity:
        return False, 0.0
    
    battery = battery_max
    time = 0.0
    prev = 0
    total_dist = 0.0
    
    for v in route:
        d = dist_matrix[prev][v]
        battery -= d
        if battery < -1e-9:
            return False, 0.0
        time += d / speed
        total_dist += d
        
        if v not in stations:
            time = max(time, ready[v])
            if time > due[v] + 1e-9:
                return False, 0.0
            time += service[v]
        else:
            energy = battery_max - battery
            time += energy * recharge_rate
            battery = battery_max
        prev = v
    
    d = dist_matrix[prev][0]
    battery -= d
    if battery < -1e-9:
        return False, 0.0
    time += d / speed
    total_dist += d
    
    if time > horizon + 1e-9:
        return False, 0.0
    
    return True, total_dist

def _construct_greedy_routes(instance, dist_matrix):
    """
    Construct routes using Greedy Nearest Neighbor.
    1. Find the 'start' customer for a new route (e.g., farthest from depot).
    2. Extend the route by adding the nearest feasible customer.
    3. Repeat until all customers are served.
    """
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    unserved = set(customers)
    routes = []
    
    # Sort customers by distance from depot to pick a good starting point for each new route
    # Or just pick the one with the earliest due date to handle time windows better?
    # Let's pick the one farthest from depot to minimize the number of routes (longer routes).
    # Actually, for TSP-like problems, starting with the farthest often helps.
    # Let's try starting with the customer with the smallest due date to ensure tight windows are met.
    # But we need to balance. Let's just iterate through unserved and pick the one with the largest distance to depot as the seed.
    
    while unserved:
        # Pick a seed: farthest from depot among unserved
        seed = max(unserved, key=lambda c: dist_matrix[0][c])
        current_route = [seed]
        unserved.remove(seed)
        
        # Try to extend the route
        while True:
            # Find the nearest feasible customer to the end of the current route
            last_node = current_route[-1]
            best_candidate = -1
            best_dist = float('inf')
            
            for c in unserved:
                d = dist_matrix[last_node][c]
                if d < best_dist:
                    # Check if adding c is feasible
                    test_route = current_route + [c]
                    feas, _ = _simulate_route(test_route, instance, dist_matrix)
                    if feas:
                        best_dist = d
                        best_candidate = c
            
            if best_candidate == -1:
                break
            
            current_route.append(best_candidate)
            unserved.remove(best_candidate)
        
        routes.append(current_route)
    
    return routes

def _two_opt(route, instance, dist_matrix, max_iter=50):
    """Apply 2-opt within a route."""
    if len(route) < 2:
        return route
        
    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        for i in range(len(route) - 1):
            for j in range(i + 1, len(route)):
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                if new_route == route:
                    continue
                feas, new_dist = _simulate_route(new_route, instance, dist_matrix)
                if feas:
                    _, old_dist = _simulate_route(route, instance, dist_matrix)
                    if new_dist < old_dist - 1e-9:
                        route = new_route
                        improved = True
                        break
            if improved:
                break
    return route

def _or_opt(route, instance, dist_matrix, max_iter=50):
    """Apply Or-opt moves (move segment of length 1, 2, or 3)."""
    if len(route) < 2:
        return route
        
    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        for seg_len in [1, 2, 3]:
            if len(route) < seg_len:
                continue
            for i in range(len(route) - seg_len + 1):
                segment = route[i:i+seg_len]
                # Remove segment
                remainder = route[:i] + route[i+seg_len:]
                if not remainder:
                    continue
                # Try inserting segment at all positions in remainder
                for j in range(len(remainder) + 1):
                    new_route = remainder[:j] + segment + remainder[j:]
                    if new_route == route:
                        continue
                    feas, new_dist = _simulate_route(new_route, instance, dist_matrix)
                    if feas:
                        _, old_dist = _simulate_route(route, instance, dist_matrix)
                        if new_dist < old_dist - 1e-9:
                            route = new_route
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    return route

def _reinsertion(routes, instance, dist_matrix, max_iter=10):
    """
    Remove a customer from one route and try to reinsert it into the best position
    in any route (including the original one).
    """
    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        for r_idx in range(len(routes)):
            for pos in range(len(routes[r_idx])):
                c = routes[r_idx][pos]
                # Remove c
                route_without = routes[r_idx][:pos] + routes[r_idx][pos+1:]
                
                # Try reinserting into all routes
                best_gain = 0.0
                best_r_idx = -1
                best_pos = -1
                
                # Calculate current total cost
                current_total_cost = 0.0
                for r in routes:
                    feas, cost = _simulate_route(r, instance, dist_matrix)
                    if feas:
                        current_total_cost += cost
                    else:
                        current_total_cost = float('inf')
                
                if current_total_cost == float('inf'):
                    break # Should not happen if we start feasible

                for r2_idx in range(len(routes)):
                    r2 = routes[r2_idx]
                    # If we removed from r_idx, and r2_idx == r_idx, the base route is route_without
                    # Otherwise, the base route is r2
                    base_route = route_without if r2_idx == r_idx else r2
                    
                    # Try inserting c at all positions in base_route
                    for ins_pos in range(len(base_route) + 1):
                        new_r2 = base_route[:ins_pos] + [c] + base_route[ins_pos:]
                        
                        # We need to check feasibility of new_r2
                        # And if r2_idx != r_idx, we also need to check route_without (which is just the old r_idx without c)
                        # Actually, we are comparing the total cost of the system.
                        # Let's just check the new route cost and the cost of the other routes (which remain unchanged except r_idx).
                        
                        feas_new, cost_new = _simulate_route(new_r2, instance, dist_matrix)
                        if not feas_new:
                            continue
                        
                        # Calculate cost of the rest of the system
                        cost_rest = 0.0
                        for r3_idx in range(len(routes)):
                            if r3_idx == r2_idx:
                                continue # We already calculated new_r2
                            if r3_idx == r_idx:
                                # This is the route we removed from. If r2_idx != r_idx, this route is now route_without
                                # If r2_idx == r_idx, this route IS new_r2, so we skip it in the 'rest' loop?
                                # Wait, if r2_idx == r_idx, we are reinserting into the same route.
                                # The 'rest' is all other routes.
                                # If r2_idx != r_idx, the 'rest' includes route_without (for r_idx) and all other routes except r2_idx.
                                pass
                        
                        # This logic is getting complex. Let's simplify:
                        # Construct the full new set of routes and calculate total cost.
                        # Since N is small (25 customers), this is affordable.
                        
                        new_routes = [r[:] for r in routes]
                        new_routes[r_idx] = route_without
                        new_routes[r2_idx] = new_r2
                        
                        total_cost = 0.0
                        feasible = True
                        for r in new_routes:
                            if not r: continue
                            feas, cost = _simulate_route(r, instance, dist_matrix)
                            if not feas:
                                feasible = False
                                break
                            total_cost += cost
                            
                        if not feasible:
                            continue
                        
                        if total_cost < current_total_cost - 1e-9:
                            gain = current_total_cost - total_cost
                            if gain > best_gain:
                                best_gain = gain
                                best_r_idx = r2_idx
                                best_pos = ins_pos
                                best_new_r2 = new_r2
                                best_route_without = route_without
                
                if best_r_idx != -1:
                    routes[r_idx] = best_route_without
                    routes[best_r_idx] = best_new_r2
                    improved = True
                    break
            if improved:
                break
    return routes

def plan(instance):
    coords = np.array(instance["coords"], dtype=float)
    dist_matrix = _precompute_dist_matrix(coords)
    
    # Build initial routes using Greedy Nearest Neighbor
    routes = _construct_greedy_routes(instance, dist_matrix)
    
    # Ensure all customers are served
    served = set()
    for r in routes:
        for v in r:
            if v not in set(instance["stations"]):
                served.add(v)
    missing = set(range(1, instance["n_customers"] + 1)) - served
    for v in missing:
        routes.append([v])
        
    # Local Search
    # 1. 2-Opt and Or-Opt on each route
    for _ in range(5): # Multiple passes
        for i in range(len(routes)):
            routes[i] = _two_opt(routes[i], instance, dist_matrix)
            routes[i] = _or_opt(routes[i], instance, dist_matrix)
        
        # 2. Reinsertion (Move customers between/within routes)
        routes = _reinsertion(routes, instance, dist_matrix)
        
        # Remove empty routes
        routes = [r for r in routes if r]
        
        if not routes:
            break
            
    # Verify all customers served exactly once
    all_nodes = []
    for r in routes:
        all_nodes.extend(r)
    customer_nodes = [v for v in all_nodes if v not in set(instance["stations"])]
    if sorted(customer_nodes) != list(range(1, instance["n_customers"] + 1)):
        # Fallback: one per route
        customers = list(range(1, instance["n_customers"] + 1))
        customers.sort(key=lambda v: instance["due"][v])
        routes = [[v] for v in customers]
        
    return routes
# EVOLVE-BLOCK-END
