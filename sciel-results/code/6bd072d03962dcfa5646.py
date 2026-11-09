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
            node_type[i] = 1  # customer
        elif i in stations:
            node_type[i] = 2  # station
    
    customer_ids = list(range(1, n_customers + 1))
    
    # Precompute useful arrays
    is_customer = np.zeros(n_nodes, dtype=bool)
    is_customer[1:n_customers+1] = True
    is_station = np.zeros(n_nodes, dtype=bool)
    for s in stations:
        is_station[s] = True
    
    # Build initial solution using nearest neighbor heuristic
    def build_initial_routes():
        """Build initial routes using nearest neighbor with capacity and battery checks."""
        unvisited = set(customer_ids)
        routes = []
        
        while unvisited:
            route = []
            total_demand = 0
            current_node = depot
            current_time = 0.0
            current_battery = battery_max
            
            while unvisited:
                # Find nearest feasible customer
                best_customer = None
                best_dist = float('inf')
                best_time = 0.0
                best_battery = 0.0
                
                for cust in unvisited:
                    d = dist[current_node, cust]
                    if d > current_battery:
                        continue
                    if total_demand + demands[cust] > capacity:
                        continue
                    
                    new_battery = current_battery - d
                    new_time = current_time + d / speed
                    
                    # Check if we need to charge before reaching customer
                    # We can't charge mid-route in this simple check, so skip for now
                    # The simulation will handle charging properly
                    
                    if new_time < ready[cust]:
                        new_time = ready[cust]
                    if new_time > due[cust]:
                        continue
                    new_time += service[cust]
                    
                    # Check if we can return to depot from this customer
                    d_return = dist[cust, depot]
                    if new_battery < d_return:
                        continue
                    
                    if d < best_dist:
                        best_dist = d
                        best_customer = cust
                        best_time = new_time
                        best_battery = new_battery
                
                if best_customer is None:
                    break
                
                route.append(best_customer)
                unvisited.remove(best_customer)
                total_demand += demands[best_customer]
                current_node = best_customer
                current_time = best_time
                current_battery = best_battery
            
            if route:
                routes.append(route)
        
        # Ensure all customers are visited
        if unvisited:
            for cust in unvisited:
                routes.append([cust])
        
        return routes
    
    routes = build_initial_routes()
    
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
    
    # Validate initial solution
    if not is_valid_solution(routes):
        # Fallback: one customer per route
        routes = [[v] for v in customer_ids]
    
    best_routes = [list(r) for r in routes]
    
    # Local search
    start_time = time.time()
    time_limit = 110.0
    
    improved = True
    no_improve_count = 0
    max_no_improve = 5
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        n_routes = len(routes)
        
        # 2-opt within each route
        for r in range(n_routes):
            route = routes[r]
            L = len(route)
            for i in range(L - 1):
                if (time.time() - start_time) > time_limit:
                    break
                for j in range(i + 1, L):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    # Check capacity
                    cap = sum(demands[node] for node in new_route)
                    if cap > capacity:
                        continue
                    d, feasible, _, _ = simulate_route(new_route)
                    if feasible:
                        # Calculate total distance change
                        old_d, _, _, _ = simulate_route(route)
                        if d < old_d:
                            routes[r] = new_route
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
                    
                    for r2 in range(n_routes):
                        if r1 == r2 and seg_start == 0 and seg_end == L1 - 1:
                            continue
                        route2 = routes[r2]
                        L2 = len(route2)
                        
                        for pos in range(L2 + 1):
                            # Check capacity
                            cap2 = sum(demands[node] for node in route2) + sum(demands[node] for node in segment)
                            if cap2 > capacity:
                                continue
                            
                            new_route2 = route2[:pos] + segment + route2[pos:]
                            d2, f2, _, _ = simulate_route(new_route2)
                            if not f2:
                                continue
                            
                            if r1 == r2:
                                # Intra-route relocation
                                new_route1 = remaining1
                            else:
                                new_route1 = remaining1
                            
                            d1, f1, _, _ = simulate_route(new_route1)
                            if not f1:
                                continue
                            
                            old_d1, _, _, _ = simulate_route(route1)
                            old_d2, _, _, _ = simulate_route(route2)
                            if d1 + d2 < old_d1 + old_d2:
                                routes[r1] = new_route1
                                if r1 != r2:
                                    routes[r2] = new_route2
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
                        # Check capacity
                        cap1 = sum(demands[node] for node in routes[r1]) - demands[n1] + demands[n2]
                        cap2 = sum(demands[node] for node in routes[r2]) - demands[n2] + demands[n1]
                        if cap1 > capacity or cap2 > capacity:
                            continue
                        new_r1 = list(routes[r1])
                        new_r2 = list(routes[r2])
                        new_r1[i1] = n2
                        new_r2[i2] = n1
                        d1, f1, _, _ = simulate_route(new_r1)
                        d2, f2, _, _ = simulate_route(new_r2)
                        if f1 and f2:
                            old_d1, _, _, _ = simulate_route(routes[r1])
                            old_d2, _, _, _ = simulate_route(routes[r2])
                            if d1 + d2 < old_d1 + old_d2:
                                routes[r1] = new_r1
                                routes[r2] = new_r2
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
