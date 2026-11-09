import numpy as np
import time
import math

def plan(instance):
    start_time = time.time()
    deadline = 145.0 # 5s margin
    
    coords = np.array(instance["coords"])
    n_nodes = len(coords)
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    stations = instance["stations"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    # Precompute distance matrix
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            d = math.hypot(coords[i][0]-coords[j][0], coords[i][1]-coords[j][1])
            dist[i][j] = d
            dist[j][i] = d

    def get_dist(i, j):
        return dist[i][j]

    def simulate_route(route):
        """
        Simulates a route (list of nodes, excluding depot start/end).
        Returns (total_distance, total_time, is_feasible)
        """
        if not route:
            return 0.0, 0.0, True
            
        curr_node = 0
        curr_time = 0.0
        curr_bat = float(battery_cap)
        total_dist = 0.0
        total_demand = 0
        
        for node in route:
            d = get_dist(curr_node, node)
            curr_time += d / speed
            curr_bat -= d
            total_dist += d
            
            if curr_bat < 0:
                return total_dist, curr_time, False
            
            # Arrive at node
            if node in stations:
                # Charge
                charge_amount = battery_cap - curr_bat
                if charge_amount > 0:
                    charge_time = charge_amount * recharge_rate
                    curr_time += charge_time
                    curr_bat = battery_cap
            else:
                # Customer
                if node <= n_customers:
                    total_demand += demands[node]
                    if total_demand > capacity:
                        return total_dist, curr_time, False
                    
                    # Time window
                    if curr_time < ready[node]:
                        curr_time = ready[node]
                    curr_time += service[node]
                    
                    if curr_time > due[node]:
                        return total_dist, curr_time, False
            
            curr_node = node
            
        # Return to depot
        d = get_dist(curr_node, 0)
        curr_time += d / speed
        curr_bat -= d
        total_dist += d
        
        if curr_bat < 0:
            return total_dist, curr_time, False
        if curr_time > horizon:
            return total_dist, curr_time, False
            
        return total_dist, curr_time, True

    def is_feasible(route):
        _, _, feas = simulate_route(route)
        return feas

    # Initial Construction: Greedy Insertion
    # Sort customers by farthest from depot to encourage clustering
    cust_dist_from_depot = [get_dist(0, c) for c in customers]
    sorted_customers = [c for _, c in sorted(zip(cust_dist_from_depot, customers), reverse=True)]
    
    routes = []
    unvisited = set(sorted_customers)
    
    def try_insert_into_route(route, cust):
        """
        Tries to insert cust into route.
        Returns (new_route, delta_dist) or None.
        """
        if not route:
            # New route: [cust]
            new_route = [cust]
            d_dist, d_time, feas = simulate_route(new_route)
            if feas:
                return new_route, d_dist
            return None
            
        best_new_route = None
        best_delta = float('inf')
        
        # Try all positions
        for i in range(len(route) + 1):
            # 1. Try without station
            new_route = route[:i] + [cust] + route[i:]
            d_dist, d_time, feas = simulate_route(new_route)
            if feas:
                # Calculate delta distance roughly for comparison
                # Exact delta is hard, but we can compare total dist
                if d_dist < best_delta:
                    best_delta = d_dist
                    best_new_route = new_route
            
            # 2. If not feasible, try inserting a station
            if not feas:
                # Find where battery failed? 
                # Simpler: Try inserting each station after the last valid point?
                # Or just try inserting each station at each position in the new route?
                # This is expensive. 
                # Heuristic: If battery fails, we likely need a station before the failing node.
                # Let's try inserting a station right before the customer or before the failing segment.
                
                # Identify the segment where battery < 0.
                # We can just try inserting each station at each position in `new_route`
                for s in stations:
                    if s in new_route: continue
                    for j in range(len(new_route) + 1):
                        candidate = new_route[:j] + [s] + new_route[j:]
                        c_dist, c_time, c_feas = simulate_route(candidate)
                        if c_feas and c_dist < best_delta:
                            best_delta = c_dist
                            best_new_route = candidate

        if best_new_route:
            return best_new_route, best_delta
        return None

    for cust in sorted_customers:
        if time.time() > deadline:
            break
        best_route_idx = -1
        best_delta = float('inf')
        best_new_route = None
        
        for idx, route in enumerate(routes):
            res = try_insert_into_route(route, cust)
            if res:
                new_route, d_dist = res
                # We want to minimize total distance, so we minimize the new route's distance
                # But we also need to consider the number of routes? 
                # Usually, minimizing distance per route is good.
                if d_dist < best_delta:
                    best_delta = d_dist
                    best_route_idx = idx
                    best_new_route = new_route
        
        if best_route_idx != -1:
            routes[best_route_idx] = best_new_route
            unvisited.remove(cust)
        else:
            # Start new route
            routes.append([cust])
            unvisited.remove(cust)
            
    # Ensure all customers are visited
    # If some were skipped due to deadline, add them individually if possible
    for cust in list(unvisited):
        routes.append([cust])
        unvisited.remove(cust)
        
    # Local Search
    def get_total_distance(routes):
        total = 0
        for r in routes:
            d, _, _ = simulate_route(r)
            total += d
        return total

    current_dist = get_total_distance(routes)
    
    improved = True
    while improved and (time.time() < deadline):
        improved = False
        
        # 2-opt Intra-route
        for r_idx in range(len(routes)):
            route = routes[r_idx]
            n = len(route)
            if n < 2: continue
            
            for i in range(n - 1):
                for j in range(i + 1, n):
                    # Reverse segment i..j
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if new_route == route: continue
                    
                    d_dist, d_time, feas = simulate_route(new_route)
                    if feas:
                        # Check if distance improved
                        # We need old distance of this route
                        old_d, _, _ = simulate_route(route)
                        if d_dist < old_d - 1e-6:
                            routes[r_idx] = new_route
                            current_dist = get_total_distance(routes)
                            improved = True
                            break
                if improved: break
            if improved: break
        if improved: continue

        # Relocate Inter-route
        for r_idx in range(len(routes)):
            if time.time() > deadline: break
            route = routes[r_idx]
            for i in range(len(route)):
                cust = route[i]
                # Remove from current route
                temp_route = route[:i] + route[i+1:]
                d_old, _, feas_old = simulate_route(temp_route)
                if not feas_old:
                    continue # Can't remove if it breaks feasibility (e.g. battery chain)
                
                # Try inserting into other routes
                for o_idx in range(len(routes)):
                    if o_idx == r_idx: continue
                    target_route = routes[o_idx]
                    res = try_insert_into_route(target_route, cust)
                    if res:
                        new_target_route, d_new_target = res
                        # Check if total distance improves
                        # Old total: d_old (source) + old_target_dist
                        # New total: d_new_target (target) + d_old (source)
                        # Wait, d_old is the distance of source WITHOUT cust.
                        # We need distance of source WITH cust (original) to compare?
                        # No, we are moving cust.
                        # Current State: Route A has cust. Route B does not.
                        # New State: Route A has no cust. Route B has cust.
                        
                        d_source_orig, _, _ = simulate_route(route)
                        d_target_orig, _, _ = simulate_route(target_route)
                        
                        d_source_new, _, _ = simulate_route(temp_route)
                        d_target_new, _, _ = simulate_route(new_target_route)
                        
                        if d_source_new + d_target_new < d_source_orig + d_target_orig - 1e-6:
                            routes[r_idx] = temp_route
                            routes[o_idx] = new_target_route
                            current_dist = get_total_distance(routes)
                            improved = True
                            break
                if improved: break
            if improved: break
        if improved: continue

        # Swap Inter-route
        for r_idx in range(len(routes)):
            if time.time() > deadline: break
            route = routes[r_idx]
            for i in range(len(route)):
                cust1 = route[i]
                for o_idx in range(r_idx + 1, len(routes)):
                    if time.time() > deadline: break
                    target_route = routes[o_idx]
                    for j in range(len(target_route)):
                        cust2 = target_route[j]
                        
                        new_route1 = route[:i] + [cust2] + route[i+1:]
                        new_route2 = target_route[:j] + [cust1] + target_route[j+1:]
                        
                        d1, _, f1 = simulate_route(new_route1)
                        d2, _, f2 = simulate_route(new_route2)
                        
                        if f1 and f2:
                            old_d1, _, _ = simulate_route(route)
                            old_d2, _, _ = simulate_route(target_route)
                            
                            if d1 + d2 < old_d1 + old_d2 - 1e-6:
                                routes[r_idx] = new_route1
                                routes[o_idx] = new_route2
                                current_dist = get_total_distance(routes)
                                improved = True
                                break
                    if improved: break
                if improved: break
            if improved: break

    return routes
