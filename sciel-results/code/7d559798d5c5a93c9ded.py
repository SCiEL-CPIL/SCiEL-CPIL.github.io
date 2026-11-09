"""EV routing with time windows, charging stations, and battery constraints."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import time

def plan(instance):
    coords = np.array(instance["coords"], dtype=np.float64)
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    n_nodes = len(coords)
    depot = 0
    
    # Precompute distance matrix
    dist = cdist(coords, coords)
    
    # Node types: 0 = depot, 1 = customer, 2 = station
    node_type = np.zeros(n_nodes, dtype=np.int8)
    for i in range(1, n_nodes):
        if i <= n_customers:
            node_type[i] = 1
        elif i in stations:
            node_type[i] = 2
    
    customer_ids = list(range(1, n_customers + 1))
    
    # Precompute useful arrays
    is_customer = np.zeros(n_nodes, dtype=bool)
    is_customer[1:n_customers+1] = True
    is_station = np.zeros(n_nodes, dtype=bool)
    for s in stations:
        is_station[s] = True
    
    # Precompute distance to depot for each node
    dist_to_depot = dist[:, depot]
    
    def simulate_route(route):
        """Simulate a route, return (total_distance, feasible, end_time, end_battery)"""
        if not route:
            return 0.0, True, 0.0, battery_max
        total_dist = 0.0
        t = 0.0
        batt = battery_max
        prev = depot
        
        for node in route:
            d = dist[prev, node]
            if d > batt:
                return total_dist + d, False, 0, 0
            batt -= d
            t += d / speed
            total_dist += d
            
            if node_type[node] == 1:  # customer
                if t < ready[node]:
                    t = ready[node]
                if t > due[node]:
                    return total_dist, False, 0, 0
                t += service[node]
            elif node_type[node] == 2:  # station
                t += (battery_max - batt) / recharge_rate
                batt = battery_max
            
            prev = node
        
        # Return to depot
        d = dist[prev, depot]
        if d > batt:
            return total_dist + d, False, 0, 0
        batt -= d
        t += d / speed
        total_dist += d
        
        if t > horizon:
            return total_dist, False, 0, 0
        
        return total_dist, True, t, batt
    
    def is_valid_solution(routes):
        """Check if all customers are served exactly once and all routes are feasible."""
        served = set()
        for route in routes:
            for node in route:
                if node_type[node] != 1:
                    return False
                if node in served:
                    return False
                served.add(node)
        if served != set(customer_ids):
            return False
        for route in routes:
            _, feasible, _, _ = simulate_route(route)
            if not feasible:
                return False
        return True
    
    # Build initial solution using a smarter heuristic
    def build_initial_routes():
        """Build initial routes using nearest neighbor with proper feasibility checks."""
        unvisited = set(customer_ids)
        routes = []
        
        # Sort customers by due date for initial ordering
        sorted_customers = sorted(customer_ids, key=lambda c: due[c])
        
        while unvisited:
            route = []
            total_demand = 0
            current_node = depot
            current_time = 0.0
            current_battery = battery_max
            
            # Try to build a route by greedily adding customers
            while unvisited:
                best_customer = None
                best_score = float('inf')
                
                for cust in unvisited:
                    if total_demand + demands[cust] > capacity:
                        continue
                    
                    d = dist[current_node, cust]
                    if d > current_battery:
                        continue
                    
                    new_battery = current_battery - d
                    new_time = current_time + d / speed
                    
                    if new_time < ready[cust]:
                        new_time = ready[cust]
                    if new_time > due[cust]:
                        continue
                    new_time += service[cust]
                    
                    # Check if we can return to depot
                    d_return = dist[cust, depot]
                    if new_battery < d_return:
                        continue
                    
                    # Score: prefer customers that are close and have tight windows
                    score = d + max(0, due[cust] - new_time) * 0.1
                    if score < best_score:
                        best_score = score
                        best_customer = cust
                
                if best_customer is None:
                    break
                
                route.append(best_customer)
                unvisited.remove(best_customer)
                total_demand += demands[best_customer]
                current_node = best_customer
                current_time = best_score  # This is approximate; we'll re-simulate later
                # Actually, let's track time properly
                d = dist[route[-2] if len(route) > 1 else depot, best_customer]
                current_time = (current_time - service[best_customer]) + d / speed
                if current_time < ready[best_customer]:
                    current_time = ready[best_customer]
                current_time += service[best_customer]
                current_battery -= d
            
            if route:
                routes.append(route)
        
        # Ensure all customers are visited
        if unvisited:
            for cust in sorted(unvisited, key=lambda c: due[c]):
                routes.append([cust])
        
        return routes
    
    routes = build_initial_routes()
    
    # Validate initial solution
    if not is_valid_solution(routes):
        # Fallback: one customer per route
        routes = [[v] for v in customer_ids]
    
    best_routes = [list(r) for r in routes]
    
    # Local search with incremental distance computation
    start_time = time.time()
    time_limit = 110.0
    
    def get_route_distance(route):
        """Compute total distance of a route including depot legs."""
        if not route:
            return 0.0
        total = dist[depot, route[0]]
        for i in range(len(route) - 1):
            total += dist[route[i], route[i+1]]
        total += dist[route[-1], depot]
        return total
    
    def fast_feasible(route):
        """Quick feasibility check - checks battery and time windows."""
        if not route:
            return True
        t = 0.0
        batt = battery_max
        prev = depot
        
        for node in route:
            d = dist[prev, node]
            if d > batt:
                return False
            batt -= d
            t += d / speed
            
            if node_type[node] == 1:
                if t < ready[node]:
                    t = ready[node]
                if t > due[node]:
                    return False
                t += service[node]
            elif node_type[node] == 2:
                t += (battery_max - batt) / recharge_rate
                batt = battery_max
            
            prev = node
        
        d = dist[prev, depot]
        if d > batt:
            return False
        t += d / speed
        
        if t > horizon:
            return False
        
        return True
    
    # Precompute route distances and demands
    route_dists = [get_route_distance(r) for r in routes]
    route_demands = [sum(demands[node] for node in r) for r in routes]
    
    improved = True
    no_improve_count = 0
    max_no_improve = 3
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        n_routes = len(routes)
        
        # 2-opt within each route
        for r in range(n_routes):
            route = routes[r]
            L = len(route)
            if L < 2:
                continue
            for i in range(L - 1):
                if (time.time() - start_time) > time_limit:
                    break
                for j in range(i + 1, L):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if route_demands[r] > capacity:
                        continue
                    if not fast_feasible(new_route):
                        continue
                    new_d = get_route_distance(new_route)
                    if new_d < route_dists[r] - 1e-9:
                        routes[r] = new_route
                        route_dists[r] = new_d
                        improved = True
        
        # Or-opt: relocate a segment of 1-2 customers to another route
        for r1 in range(n_routes):
            if (time.time() - start_time) > time_limit:
                break
            route1 = routes[r1]
            L1 = len(route1)
            for seg_start in range(L1):
                for seg_len in [1, 2]:
                    seg_end = seg_start + seg_len - 1
                    if seg_end >= L1:
                        continue
                    segment = route1[seg_start:seg_end+1]
                    remaining1 = route1[:seg_start] + route1[seg_end+1:]
                    seg_demand = sum(demands[node] for node in segment)
                    
                    for r2 in range(n_routes):
                        if r1 == r2 and seg_start == 0 and seg_end == L1 - 1:
                            continue
                        route2 = routes[r2]
                        L2 = len(route2)
                        
                        if route_demands[r2] + seg_demand > capacity:
                            continue
                        
                        for pos in range(L2 + 1):
                            new_route2 = route2[:pos] + segment + route2[pos:]
                            if not fast_feasible(new_route2):
                                continue
                            
                            if r1 == r2:
                                new_route1 = remaining1
                            else:
                                new_route1 = remaining1
                            
                            if not fast_feasible(new_route1):
                                continue
                            
                            d1 = get_route_distance(new_route1)
                            d2 = get_route_distance(new_route2)
                            if d1 + d2 < route_dists[r1] + route_dists[r2] - 1e-9:
                                routes[r1] = new_route1
                                route_dists[r1] = d1
                                route_demands[r1] = sum(demands[node] for node in new_route1)
                                if r1 != r2:
                                    routes[r2] = new_route2
                                    route_dists[r2] = d2
                                    route_demands[r2] = sum(demands[node] for node in new_route2)
                                improved = True
        
        # Swap between routes
        for r1 in range(n_routes):
            if (time.time() - start_time) > time_limit:
                break
            for r2 in range(r1 + 1, n_routes):
                L1 = len(routes[r1])
                L2 = len(routes[r2])
                for i1 in range(L1):
                    for i2 in range(L2):
                        n1 = routes[r1][i1]
                        n2 = routes[r2][i2]
                        cap1 = route_demands[r1] - demands[n1] + demands[n2]
                        cap2 = route_demands[r2] - demands[n2] + demands[n1]
                        if cap1 > capacity or cap2 > capacity:
                            continue
                        
                        new_r1 = list(routes[r1])
                        new_r2 = list(routes[r2])
                        new_r1[i1] = n2
                        new_r2[i2] = n1
                        
                        if fast_feasible(new_r1) and fast_feasible(new_r2):
                            d1 = get_route_distance(new_r1)
                            d2 = get_route_distance(new_r2)
                            if d1 + d2 < route_dists[r1] + route_dists[r2] - 1e-9:
                                routes[r1] = new_r1
                                routes[r2] = new_r2
                                route_dists[r1] = d1
                                route_dists[r2] = d2
                                route_demands[r1] = cap1
                                route_demands[r2] = cap2
                                improved = True
        
        if not improved:
            no_improve_count += 1
        else:
            no_improve_count = 0
        
        if no_improve_count >= max_no_improve:
            break
    
    # Final validation
    if is_valid_solution(routes):
        return routes
    else:
        return [[v] for v in customer_ids]
# EVOLVE-BLOCK-END
