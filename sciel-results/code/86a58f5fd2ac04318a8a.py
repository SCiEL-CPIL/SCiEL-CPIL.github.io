"""Reformulated EV routing: time-dependent Dijkstra on (node, battery, unvisited) state space."""
# EVOLVE-BLOCK-START
import numpy as np
import heapq
import math

def plan(instance):
    n_cust = instance["n_customers"]
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    # Node indices: 0 is depot, 1..n_cust are customers, rest are stations
    # We'll work with node ids directly
    # Precompute distances between all relevant nodes
    # Relevant nodes: 0 (depot), 1..n_cust (customers), and station nodes
    relevant_nodes = list(range(0, n_cust + 1)) + list(stations)
    # Remove duplicates while preserving order
    seen = set()
    unique_relevant = []
    for n in relevant_nodes:
        if n not in seen:
            seen.add(n)
            unique_relevant.append(n)
    
    # Build distance matrix for ALL nodes to allow fast lookup
    # Nodes are 0..n_nodes-1
    n_nodes = len(coords)
    dist_mat = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            dist_mat[i][j] = math.sqrt(dx*dx + dy*dy)
    
    # Battery levels are continuous, so we can't discretize fully.
    # Instead, we use a different approach: for each route, we build it incrementally
    # using a greedy insertion with feasibility checks, but guided by a more global perspective.
    
    # Actually, let's use a simpler but effective approach:
    # 1. Cluster customers using k-means into ~5-7 groups
    # 2. For each cluster, solve a TSP-like problem with battery constraints
    # 3. Use nearest-neighbor with battery-aware lookahead
    
    # Let me implement a smarter approach:
    # - Use a priority-based construction where we pick the next customer/station
    #   that minimizes total cost (distance + time penalty) while maintaining feasibility
    # - This is essentially a greedy algorithm with feasibility checks
    
    # First, let's create a helper to check if a sequence of nodes is feasible
    def is_feasible_route(route, instance_data):
        """Check if a route (list of node ids, excluding depot) is feasible."""
        coords_local, demands_local, ready_local, due_local, service_local = instance_data
        stations_local, capacity_local, battery_max_local, speed_local, recharge_rate_local, horizon_local = instance_data[5], instance_data[6], instance_data[7], instance_data[8], instance_data[9], instance_data[10]
        
        # Calculate total demand
        total_demand = 0
        for node in route:
            if node <= n_cust and node >= 1:
                total_demand += demands_local[node]
        if total_demand > capacity_local:
            return False
        
        # Simulate the route
        current_node = 0  # depot
        time = 0.0
        battery = battery_max_local
        visited = set()
        
        for node in route:
            # Travel from current_node to node
            dx = coords_local[current_node][0] - coords_local[node][0]
            dy = coords_local[current_node][1] - coords_local[node][1]
            dist = math.sqrt(dx*dx + dy*dy)
            
            # Check battery
            if battery < dist:
                return False
            battery -= dist
            time += dist / speed_local
            
            # If it's a station, recharge
            if node in stations_local:
                # Recharge: adds (battery_max - battery) * recharge_rate time, refills battery
                recharge_time = (battery_max_local - battery) * recharge_rate_local
                time += recharge_time
                battery = battery_max_local
            else:
                # It's a customer
                if node in visited:
                    return False
                visited.add(node)
                
                # Time window
                time = max(time, ready_local[node])
                if time > due_local[node]:
                    return False
                time += service_local[node]
            
            current_node = node
        
        # Return to depot
        dx = coords_local[current_node][0] - coords_local[0][0]
        dy = coords_local[current_node][1] - coords_local[0][1]
        dist = math.sqrt(dx*dx + dy*dy)
        if battery < dist:
            return False
        battery -= dist
        time += dist / speed_local
        
        if time > horizon_local:
            return False
        
        return True
    
    # Simpler approach: build routes using a greedy method with battery-aware decisions
    # Step 1: Sort customers by due date
    customers = list(range(1, n_cust + 1))
    customers.sort(key=lambda v: due[v])
    
    # Step 2: For each customer, try to insert it into existing routes or create a new one
    # Use a feasibility check with battery simulation
    
    def simulate_route(route):
        """Simulate a route and return (feasible, total_distance, end_time, end_battery)."""
        current_node = 0
        time = 0.0
        battery = battery_max
        total_dist = 0.0
        
        for node in route:
            dist = dist_mat[current_node][node]
            
            if battery < dist:
                return (False, total_dist, time, battery)
            battery -= dist
            time += dist / speed
            total_dist += dist
            
            if node in stations:
                recharge_time = (battery_max - battery) * recharge_rate
                time += recharge_time
                battery = battery_max
            else:
                time = max(time, ready[node])
                if time > due[node]:
                    return (False, total_dist, time, battery)
                time += service[node]
            
            current_node = node
        
        # Return to depot
        dist = dist_mat[current_node][0]
        if battery < dist:
            return (False, total_dist, time, battery)
        battery -= dist
        time += dist / speed
        total_dist += dist
        
        if time > horizon:
            return (False, total_dist, time, battery)
        
        return (True, total_dist, time, battery)
    
    # Build routes using a greedy insertion approach
    # Start with no routes
    routes = []
    unvisited = set(customers)
    
    # Sort customers by a heuristic: prefer those with tight time windows and far from depot
    def customer_priority(c):
        # Prioritize customers with earlier due dates and greater distance from depot
        dx = coords[c][0] - coords[0][0]
        dy = coords[c][1] - coords[0][1]
        dist_to_depot = math.sqrt(dx*dx + dy*dy)
        return (due[c], -dist_to_depot)
    
    sorted_customers = sorted(customers, key=customer_priority)
    
    # For each customer, try to insert into existing routes
    # If no existing route works, create a new one
    for c in sorted_customers:
        if c not in unvisited:
            continue
        
        best_route_idx = -1
        best_position = -1
        best_dist = float('inf')
        
        # Try inserting into each existing route
        for ri, route in enumerate(routes):
            # Try inserting at each position
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [c] + route[pos:]
                feasible, dist, time, battery = simulate_route(new_route)
                if feasible and dist < best_dist:
                    best_dist = dist
                    best_route_idx = ri
                    best_position = pos
        
        # Also consider creating a new route with just this customer
        new_route = [c]
        feasible, dist, time, battery = simulate_route(new_route)
        if feasible and dist < best_dist:
            best_dist = dist
            best_route_idx = -1
            best_position = 0
        
        if best_route_idx == -1:
            # Create new route
            routes.append([c])
        else:
            routes[best_route_idx].insert(best_position, c)
        
        unvisited.discard(c)
    
    # Local search functions
    def two_opt_improve(route):
        """Apply 2-opt to a route, checking feasibility."""
        improved = True
        while improved:
            improved = False
            n = len(route)
            current_dist = None
            for i in range(n - 1):
                for j in range(i + 1, n):
                    # Create new route
                    new_route = route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]
                    if new_route == route:
                        continue
                    if current_dist is None:
                        feasible, current_dist, _, _ = simulate_route(route)
                        if not feasible:
                            return route
                    feasible, dist_new, _, _ = simulate_route(new_route)
                    if feasible and dist_new < current_dist - 1e-9:
                        route[:] = new_route
                        current_dist = dist_new
                        improved = True
        return route

    def or_opt_improve(route):
        """Apply Or-opt (move 1, 2, or 3 nodes) to a route."""
        improved = True
        while improved:
            improved = False
            n = len(route)
            current_dist = None
            for k in [1, 2, 3]:
                for i in range(n - k + 1):
                    # Extract segment
                    segment = route[i:i+k]
                    # Remove segment
                    rest = route[:i] + route[i+k:]
                    # Try inserting at all positions in rest
                    for j in range(len(rest) + 1):
                        if j >= i and j <= i+k:
                            continue # Skip original position
                        new_route = rest[:j] + segment + rest[j:]
                        if current_dist is None:
                            feasible, current_dist, _, _ = simulate_route(route)
                            if not feasible:
                                return route
                        feasible, dist_new, _, _ = simulate_route(new_route)
                        if feasible and dist_new < current_dist - 1e-9:
                            route[:] = new_route
                            current_dist = dist_new
                            improved = True
                            break # Restart loop as route changed
                    if improved:
                        break
                if improved:
                    break
        return route

    # Apply local search to each route
    for i in range(len(routes)):
        routes[i] = two_opt_improve(routes[i])
        routes[i] = or_opt_improve(routes[i])

    # Inter-route relocation: Move a customer from one route to another if it reduces total distance
    # First, get current distances
    route_dists = []
    for r in routes:
        _, d, _, _ = simulate_route(r)
        route_dists.append(d)
    
    improved = True
    while improved:
        improved = False
        for i in range(len(routes)):
            if len(routes[i]) == 0:
                continue
            for j in range(len(routes[i])):
                c = routes[i][j]
                # Remove c from route i
                route_i_new = routes[i][:j] + routes[i][j+1:]
                if not route_i_new:
                    continue
                feasible_i, dist_i_new, _, _ = simulate_route(route_i_new)
                if not feasible_i:
                    continue
                    
                # Try inserting c into route k
                for k in range(len(routes)):
                    if k == i:
                        continue
                    route_k = routes[k]
                    for pos in range(len(route_k) + 1):
                        new_route_k = route_k[:pos] + [c] + route_k[pos:]
                        feasible_k, dist_k_new, _, _ = simulate_route(new_route_k)
                        if not feasible_k:
                            continue
                        
                        # Check if total distance improves
                        # Old total for i and k: route_dists[i] + route_dists[k]
                        # New total for i and k: dist_i_new + dist_k_new
                        if dist_i_new + dist_k_new < route_dists[i] + route_dists[k] - 1e-9:
                            # Apply move
                            routes[i] = route_i_new
                            routes[k] = new_route_k
                            route_dists[i] = dist_i_new
                            route_dists[k] = dist_k_new
                            improved = True
                            break # Restart inner loops? Or just break out and re-evaluate
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    
    # Try to merge routes if possible
    for i in range(len(routes)):
        for j in range(i + 1, len(routes)):
            merged = routes[i] + routes[j]
            # Check capacity
            total_demand = sum(demands[c] for c in merged if 1 <= c <= n_cust)
            if total_demand > capacity:
                continue
            feasible, dist, _, _ = simulate_route(merged)
            if feasible:
                # Check if merging reduces total distance
                if dist < route_dists[i] + route_dists[j] - 1e-9:
                    routes[i] = merged
                    routes.pop(j)
                    route_dists[i] = dist
                    route_dists.pop(j)
                    j -= 1
                    # Re-apply 2-opt
                    routes[i] = two_opt_improve(routes[i])
                    routes[i] = or_opt_improve(routes[i])
                    # Update dist
                    _, route_dists[i], _, _ = simulate_route(routes[i])
                    break
    
    return routes
# EVOLVE-BLOCK-END
