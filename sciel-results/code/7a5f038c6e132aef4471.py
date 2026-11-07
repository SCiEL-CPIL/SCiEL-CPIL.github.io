# EVOLVE-BLOCK-START
import numpy as np
from math import sqrt
import heapq

def plan(instance):
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    n_nodes = 1 + n_customers + len(stations)
    
    # Precompute distances
    D = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            D[i,j] = sqrt(dx*dx + dy*dy)
            D[j,i] = D[i,j]
    
    cust_nodes = list(range(1, n_customers + 1))
    station_nodes = list(stations)
    
    def is_customer(node):
        return 1 <= node <= n_customers
        
    def check_route(route):
        """Check if a route is feasible. Returns (feasible, total_distance)."""
        time = 0.0
        battery = battery_cap
        load = 0
        prev = 0
        total_dist = 0.0
        
        for node in route:
            d = D[prev, node]
            if battery - d < -1e-9:
                return False, 0.0
            time += d / speed
            battery -= d
            total_dist += d
            if is_customer(node):
                if time > due[node] + 1e-9:
                    return False, 0.0
                time = max(time, ready[node])
                time += service[node]
                load += demands[node]
                if load > capacity + 1e-9:
                    return False, 0.0
            elif node in station_nodes:
                # Recharge
                energy_needed = battery_cap - battery
                if energy_needed > 0:
                    time += energy_needed * recharge_rate
                battery = battery_cap
            prev = node
        
        # Return to depot
        d = D[prev, 0]
        if battery - d < -1e-9:
            return False, 0.0
        time += d / speed
        battery -= d
        total_dist += d
        
        if time > horizon + 1e-9:
            return False, 0.0
            
        return True, total_dist

    def insertion_heuristic():
        customers_sorted = sorted(cust_nodes, key=lambda v: due[v])
        routes = []
        
        for cust in customers_sorted:
            best_route_idx = -1
            best_pos = -1
            best_cost = float('inf')
            best_new_route = None
            
            for r_idx, route in enumerate(routes):
                # Try inserting cust at each position
                for pos in range(len(route) + 1):
                    # 1. Try without station
                    new_route = route[:pos] + [cust] + route[pos:]
                    feasible, cost = check_route(new_route)
                    if feasible and cost < best_cost:
                        best_cost = cost
                        best_route_idx = r_idx
                        best_pos = pos
                        best_new_route = new_route
                    
                    # 2. Try with a station inserted nearby if battery is tight
                    # We check if the trip to cust or from cust requires charging.
                    # Simple heuristic: try inserting each station at pos-1, pos, pos+1
                    # Only do this if the direct insertion failed or is close to limit?
                    # For simplicity, always try a few station insertions if direct failed.
                    if not feasible:
                        # Check if it failed due to battery or time
                        # We don't know exactly, but let's try stations.
                        # Try inserting station s at pos-1, pos, pos+1
                        for s in station_nodes:
                            # Avoid inserting station at start/end if not needed?
                            # pos-1
                            if pos - 1 >= 0:
                                nr = route[:pos-1] + [s, cust] + route[pos:]
                                f, c = check_route(nr)
                                if f and c < best_cost:
                                    best_cost = c
                                    best_route_idx = r_idx
                                    best_pos = pos # Store pos relative to original, but we need to store the actual new route
                                    best_new_route = nr
                            # pos (before cust)
                            nr = route[:pos] + [s, cust] + route[pos:]
                            f, c = check_route(nr)
                            if f and c < best_cost:
                                best_cost = c
                                best_route_idx = r_idx
                                best_pos = pos
                                best_new_route = nr
                            # pos+1 (after cust)
                            nr = route[:pos+1] + [s] + route[pos+1:] # Wait, cust is at pos in new_route logic?
                            # In new_route = route[:pos] + [cust] + route[pos:], cust is at index pos.
                            # So after cust is pos+1.
                            nr = route[:pos] + [cust, s] + route[pos:]
                            f, c = check_route(nr)
                            if f and c < best_cost:
                                best_cost = c
                                best_route_idx = r_idx
                                best_pos = pos
                                best_new_route = nr
            
            if best_new_route is not None:
                routes[best_route_idx] = best_new_route
            else:
                # Create new route
                # Try with station?
                best_new_route = None
                best_cost = float('inf')
                # Direct
                nr = [cust]
                f, c = check_route(nr)
                if f and c < best_cost:
                    best_cost = c
                    best_new_route = nr
                # With station
                for s in station_nodes:
                    nr = [s, cust]
                    f, c = check_route(nr)
                    if f and c < best_cost:
                        best_cost = c
                        best_new_route = nr
                    nr = [cust, s]
                    f, c = check_route(nr)
                    if f and c < best_cost:
                        best_cost = c
                        best_new_route = nr
                
                if best_new_route is not None:
                    routes.append(best_new_route)
                else:
                    # Should not happen, but fallback
                    routes.append([cust])
        
        return routes

    def two_opt_improve(routes):
        improved = True
        while improved:
            improved = False
            for r_idx in range(len(routes)):
                route = routes[r_idx]
                n = len(route)
                if n < 2:
                    continue
                # Calculate current cost once
                _, old_cost = check_route(route)
                
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        # Reverse segment i..j
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        feasible, cost = check_route(new_route)
                        if feasible and cost < old_cost - 1e-9:
                            routes[r_idx] = new_route
                            old_cost = cost
                            improved = True
        return routes

    def or_opt_improve(routes):
        improved = True
        while improved:
            improved = False
            for r_idx in range(len(routes)):
                route = routes[r_idx]
                n = len(route)
                if n < 2:
                    continue
                _, old_cost = check_route(route)
                
                # Try moving 1 node
                for i in range(n):
                    node = route[i]
                    # Remove node
                    rest = route[:i] + route[i+1:]
                    # Try inserting at all positions in rest
                    for pos in range(len(rest) + 1):
                        if pos == i: continue # Skip original position
                        new_route = rest[:pos] + [node] + rest[pos:]
                        feasible, cost = check_route(new_route)
                        if feasible and cost < old_cost - 1e-9:
                            routes[r_idx] = new_route
                            old_cost = cost
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return routes

    routes = insertion_heuristic()
    routes = two_opt_improve(routes)
    routes = or_opt_improve(routes)
    routes = two_opt_improve(routes)
    
    # Ensure all customers are served
    served = set()
    for route in routes:
        for node in route:
            if is_customer(node):
                served.add(node)
    
    missing = set(cust_nodes) - served
    if missing:
        for cust in missing:
            routes.append([cust])
            
    return routes
# EVOLVE-BLOCK-END
