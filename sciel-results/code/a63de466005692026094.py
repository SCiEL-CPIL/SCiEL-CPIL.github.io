# EVOLVE-BLOCK-START
import numpy as np
import time
from math import sqrt

def _euclid(coords, i, j):
    dx = coords[i, 0] - coords[j, 0]
    dy = coords[i, 1] - coords[j, 1]
    return sqrt(dx * dx + dy * dy)

def _solve_instance(instance):
    coords = np.array(instance["coords"], dtype=float)
    n_customers = instance["n_customers"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    n_nodes = len(coords)
    
    # Precompute distances
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = _euclid(coords, i, j)
            dist[i, j] = d
            dist[j, i] = d
    
    customers = list(range(1, n_customers + 1))
    
    # Sort customers by due time for initial ordering
    customers_sorted = sorted(customers, key=lambda c: due[c])
    
    # Group customers into routes using a simple bin-packing approach
    # We'll use a greedy approach: sort by angle from depot, then cluster
    depot = 0
    # Compute angle from depot for each customer
    def angle_from_depot(c):
        dx = coords[c, 0] - coords[depot, 0]
        dy = coords[c, 1] - coords[depot, 1]
        return np.arctan2(dy, dx)
    
    # Sort by angle
    customers_by_angle = sorted(customers, key=angle_from_depot)
    
    # Try to pack customers into routes respecting capacity
    # Use a simple first-fit decreasing by demand
    customers_by_demand = sorted(customers, key=lambda c: -demands[c])
    
    routes = []
    route_loads = []
    for c in customers_by_demand:
        placed = False
        for i, route in enumerate(routes):
            if route_loads[i] + demands[c] <= capacity:
                routes[i].append(c)
                route_loads[i] += demands[c]
                placed = True
                break
        if not placed:
            routes.append([c])
            route_loads.append(demands[c])
    
    # For each route, order customers using nearest neighbor with time window constraints
    # Then insert charging stations as needed
    
    def _build_route(route_customers):
        """Build a feasible route for the given customers, inserting charging if needed."""
        if not route_customers:
            return []
        
        # Sort by angle from depot as initial order
        route_customers = sorted(route_customers, key=angle_from_depot)
        
        # Try to build a feasible sequence
        # We'll use a greedy insertion approach
        route = []
        remaining = set(route_customers)
        current_node = depot
        current_time = 0.0
        current_battery = battery_cap
        total_demand = 0
        
        # We need to find an ordering that is feasible
        # Use nearest neighbor with feasibility check
        while remaining:
            best_c = None
            best_dist = float('inf')
            for c in remaining:
                d = dist[current_node, c]
                # Check if we can reach c with current battery
                if d <= current_battery:
                    # Check time window
                    arrival_time = current_time + d / speed
                    # We need to check if we can wait until ready[c]
                    # and if due[c] allows it
                    if arrival_time <= due[c]:
                        if d < best_dist:
                            best_dist = d
                            best_c = c
            
            if best_c is None:
                # Need to charge
                # Find nearest station
                best_station = None
                best_sd = float('inf')
                for s in stations:
                    d = dist[current_node, s]
                    if d <= current_battery and d < best_sd:
                        best_sd = d
                        best_station = s
                
                if best_station is None:
                    # Can't reach any station, this route is infeasible
                    # Return empty to signal failure
                    return None
                
                # Go to station
                current_time += best_sd / speed
                current_battery -= best_sd
                # Recharge
                energy_needed = battery_cap - current_battery
                recharge_time = energy_needed * recharge_rate
                current_time += recharge_time
                current_battery = battery_cap
                current_node = best_station
                
                # Now try again to find a customer
                continue
            
            # Go to customer
            current_time += best_dist / speed
            current_battery -= best_dist
            # Wait if needed
            if current_time < ready[best_c]:
                current_time = ready[best_c]
            # Service
            current_time += service[best_c]
            total_demand += demands[best_c]
            
            route.append(best_c)
            remaining.remove(best_c)
            current_node = best_c
        
        # Return to depot
        d_back = dist[current_node, depot]
        if d_back > current_battery:
            # Need to charge on the way back
            # Find a station on the way back
            # For simplicity, check if we can reach a station then depot
            feasible = False
            for s in stations:
                d1 = dist[current_node, s]
                d2 = dist[s, depot]
                if d1 <= current_battery:
                    t1 = current_time + d1 / speed
                    b1 = current_battery - d1
                    # Recharge
                    energy_needed = battery_cap - b1
                    t1 += energy_needed * recharge_rate
                    b1 = battery_cap
                    if d2 <= b1:
                        t2 = t1 + d2 / speed
                        if t2 <= horizon:
                            feasible = True
                            break
            
            if not feasible:
                return None
        
        current_time += d_back / speed
        if current_time > horizon:
            return None
        
        return route
    
    # Build routes
    final_routes = []
    for route_customers in routes:
        route = _build_route(route_customers)
        if route is None:
            # Route infeasible, split into single-customer routes
            for c in route_customers:
                single_route = _build_route([c])
                if single_route is not None:
                    final_routes.append(single_route)
                else:
                    # Even single customer infeasible? Should not happen with full battery
                    # but handle gracefully
                    final_routes.append([c])
        else:
            final_routes.append(route)
    
    return final_routes

def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    result = _solve_instance(instance)
    
    elapsed = time.time() - start_time
    if elapsed > 120.0:
        # Should not happen, but guard
        pass
    
    return result
# EVOLVE-BLOCK-END
