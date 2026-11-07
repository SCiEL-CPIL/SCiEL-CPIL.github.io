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
    
    if not route:
        return True, 0.0

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

def _construct_savings_routes(instance, dist_matrix):
    """
    Construct routes using Clarke-Wright Savings Algorithm.
    """
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    
    # Initialize routes: each customer is in their own route
    # routes[i] contains the nodes in route i
    routes = [[c] for c in customers]
    # node_to_route[i] = index of the route containing node i
    node_to_route = {c: i for i, c in enumerate(customers)}
    # route_end[i] = last node in route i (for quick access, though we can just use -1)
    # route_start[i] = first node in route i
    
    # Calculate savings: s(i, j) = d(0, i) + d(0, j) - d(i, j)
    # Merging i and j saves this distance.
    savings = []
    for i in range(len(customers)):
        for j in range(i + 1, len(customers)):
            c1 = customers[i]
            c2 = customers[j]
            s = dist_matrix[0][c1] + dist_matrix[0][c2] - dist_matrix[c1][c2]
            savings.append((s, i, j))
            
    # Sort savings in descending order
    savings.sort(reverse=True)
    
    # Merge routes
    # We need to ensure that the nodes being merged are at the ends of their respective routes
    # and that the merged route is feasible.
    
    for s, i, j in savings:
        r1 = node_to_route.get(customers[i])
        r2 = node_to_route.get(customers[j])
        
        if r1 is None or r2 is None:
            continue
        if r1 == r2:
            continue
            
        route1 = routes[r1]
        route2 = routes[r2]
        
        # Check if c1 is at an end of route1 and c2 is at an end of route2
        # There are 4 possible orientations for merging:
        # 1. route1 ends with c1, route2 starts with c2 -> route1 + route2
        # 2. route1 ends with c1, route2 ends with c2 -> route1 + reverse(route2)
        # 3. route1 starts with c1, route2 starts with c2 -> reverse(route1) + route2
        # 4. route1 starts with c1, route2 ends with c2 -> reverse(route1) + reverse(route2)
        
        merged = None
        
        # Check end of route1
        if route1[-1] == customers[i]:
            # c1 is at end of route1
            if route2[0] == customers[j]:
                # c2 is at start of route2
                candidate = route1 + route2
                feas, _ = _simulate_route(candidate, instance, dist_matrix)
                if feas:
                    merged = candidate
            elif route2[-1] == customers[j]:
                # c2 is at end of route2
                candidate = route1 + route2[::-1]
                feas, _ = _simulate_route(candidate, instance, dist_matrix)
                if feas:
                    merged = candidate
        elif route1[0] == customers[i]:
            # c1 is at start of route1
            if route2[0] == customers[j]:
                # c2 is at start of route2
                candidate = route1[::-1] + route2
                feas, _ = _simulate_route(candidate, instance, dist_matrix)
                if feas:
                    merged = candidate
            elif route2[-1] == customers[j]:
                # c2 is at end of route2
                candidate = route1[::-1] + route2[::-1]
                feas, _ = _simulate_route(candidate, instance, dist_matrix)
                if feas:
                    merged = candidate
        
        if merged is not None:
            # Update routes
            # Merge r2 into r1
            routes[r1] = merged
            routes[r2] = [] # Mark as empty
            
            # Update node_to_route
            for node in merged:
                node_to_route[node] = r1
                
            # Clean up empty routes? Or just keep them and filter later.
            # For efficiency, we can just leave them and filter at the end.
            
    # Filter out empty routes
    final_routes = [r for r in routes if r]
    return final_routes

def _local_search(routes, instance, dist_matrix, max_iter=20):
    """
    Apply local search: Relocate and Or-Opt.
    """
    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        
        # 1. Relocate: Move a single node from one route to another
        for r1_idx in range(len(routes)):
            if not routes[r1_idx]: continue
            for pos in range(len(routes[r1_idx])):
                node = routes[r1_idx][pos]
                
                # Remove node
                route1_without = routes[r1_idx][:pos] + routes[r1_idx][pos+1:]
                
                # Try inserting into all other routes (and back into route1 at different pos)
                best_gain = 0.0
                best_move = None # (r2_idx, ins_pos)
                
                # Calculate current cost
                current_cost = 0.0
                for r in routes:
                    if r:
                        feas, cost = _simulate_route(r, instance, dist_matrix)
                        if not feas: return routes
                        current_cost += cost
                
                for r2_idx in range(len(routes)):
                    if not routes[r2_idx] and r2_idx != r1_idx:
                        # If route is empty and not the source, we can insert here
                        # But we only have non-empty routes in our list mostly.
                        # If r2_idx == r1_idx, we are re-inserting into same route.
                        pass
                    
                    base_route = route1_without if r2_idx == r1_idx else routes[r2_idx]
                    
                    for ins_pos in range(len(base_route) + 1):
                        new_route2 = base_route[:ins_pos] + [node] + base_route[ins_pos:]
                        
                        # Check feasibility of new_route2
                        feas, cost_new2 = _simulate_route(new_route2, instance, dist_matrix)
                        if not feas:
                            continue
                        
                        # Check feasibility of route1_without (if r2_idx != r1_idx)
                        if r2_idx != r1_idx:
                            feas1, cost_new1 = _simulate_route(route1_without, instance, dist_matrix)
                            if not feas1:
                                continue
                        else:
                            cost_new1 = 0.0 # Cost is part of new_route2? No, new_route2 IS the modified route.
                            # Wait, if r2_idx == r1_idx, the route is new_route2.
                            # If r2_idx != r1_idx, the routes are new_route2 and route1_without.
                            
                        # Calculate total cost for this move
                        total_new_cost = 0.0
                        for r3_idx in range(len(routes)):
                            if r3_idx == r2_idx:
                                total_new_cost += cost_new2
                            elif r3_idx == r1_idx and r2_idx != r1_idx:
                                total_new_cost += cost_new1
                            elif r3_idx != r2_idx and r3_idx != r1_idx:
                                feas3, cost3 = _simulate_route(routes[r3_idx], instance, dist_matrix)
                                if not feas3:
                                    total_new_cost = float('inf')
                                    break
                                total_new_cost += cost3
                            # If r3_idx == r1_idx == r2_idx, we already added cost_new2.
                            # If r3_idx == r1_idx != r2_idx, we added cost_new1.
                            # If r3_idx == r2_idx != r1_idx, we added cost_new2.
                            # If r3_idx is other, we added cost3.
                            
                        if total_new_cost < current_cost - 1e-9:
                            gain = current_cost - total_new_cost
                            if gain > best_gain:
                                best_gain = gain
                                best_move = (r2_idx, ins_pos, new_route2, route1_without)
                
                if best_move:
                    r2_idx, ins_pos, new_route2, route1_without = best_move
                    routes[r2_idx] = new_route2
                    routes[r1_idx] = route1_without
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue
            
        # 2. Or-Opt: Move a segment of length 1, 2, or 3
        for seg_len in [1, 2, 3]:
            for r1_idx in range(len(routes)):
                if len(routes[r1_idx]) < seg_len: continue
                route1 = routes[r1_idx]
                
                for start in range(len(route1) - seg_len + 1):
                    segment = route1[start:start+seg_len]
                    route1_without = route1[:start] + route1[start+seg_len:]
                    
                    best_gain = 0.0
                    best_move = None
                    
                    current_cost = 0.0
                    for r in routes:
                        if r:
                            feas, cost = _simulate_route(r, instance, dist_matrix)
                            if not feas: return routes
                            current_cost += cost
                            
                    for r2_idx in range(len(routes)):
                        base_route = route1_without if r2_idx == r1_idx else routes[r2_idx]
                        
                        for ins_pos in range(len(base_route) + 1):
                            new_route2 = base_route[:ins_pos] + segment + base_route[ins_pos:]
                            
                            feas, cost_new2 = _simulate_route(new_route2, instance, dist_matrix)
                            if not feas:
                                continue
                            
                            if r2_idx != r1_idx:
                                feas1, cost_new1 = _simulate_route(route1_without, instance, dist_matrix)
                                if not feas1:
                                    continue
                            else:
                                cost_new1 = 0.0
                                
                            total_new_cost = 0.0
                            for r3_idx in range(len(routes)):
                                if r3_idx == r2_idx:
                                    total_new_cost += cost_new2
                                elif r3_idx == r1_idx and r2_idx != r1_idx:
                                    total_new_cost += cost_new1
                                elif r3_idx != r2_idx and r3_idx != r1_idx:
                                    feas3, cost3 = _simulate_route(routes[r3_idx], instance, dist_matrix)
                                    if not feas3:
                                        total_new_cost = float('inf')
                                        break
                                    total_new_cost += cost3
                                        
                            if total_new_cost < current_cost - 1e-9:
                                gain = current_cost - total_new_cost
                                if gain > best_gain:
                                    best_gain = gain
                                    best_move = (r2_idx, ins_pos, new_route2, route1_without)
                    
                    if best_move:
                        r2_idx, ins_pos, new_route2, route1_without = best_move
                        routes[r2_idx] = new_route2
                        routes[r1_idx] = route1_without
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
                
        # 3. 2-Opt (within route)
        for r_idx in range(len(routes)):
            route = routes[r_idx]
            if len(route) < 2: continue
            
            improved_2opt = True
            while improved_2opt:
                improved_2opt = False
                for i in range(len(route) - 1):
                    for j in range(i + 1, len(route)):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        if new_route == route: continue
                        feas, new_dist = _simulate_route(new_route, instance, dist_matrix)
                        if feas:
                            _, old_dist = _simulate_route(route, instance, dist_matrix)
                            if new_dist < old_dist - 1e-9:
                                routes[r_idx] = new_route
                                route = new_route
                                improved_2opt = True
                                improved = True
                                break
                    if improved_2opt:
                        break
    
    # Remove empty routes
    routes = [r for r in routes if r]
    return routes

def plan(instance):
    coords = np.array(instance["coords"], dtype=float)
    dist_matrix = _precompute_dist_matrix(coords)
    
    # Build initial routes using Clarke-Wright Savings
    routes = _construct_savings_routes(instance, dist_matrix)
    
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
    routes = _local_search(routes, instance, dist_matrix)
    
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
