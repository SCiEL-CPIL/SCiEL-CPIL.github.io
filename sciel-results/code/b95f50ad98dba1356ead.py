# EVOLVE-BLOCK-START
import numpy as np
from math import sqrt

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
    
    # Build node list: 0 is depot, 1..n_customers are customers, then stations
    # coords is a dict or list? Assume list indexed by node id.
    # From task: "node 0 the depot, customers 1..25, then stations"
    # So total nodes = 1 + n_customers + len(stations)
    
    n_nodes = 1 + n_customers + len(stations)
    
    # Precompute distances
    def dist(i, j):
        dx = coords[i][0] - coords[j][0]
        dy = coords[i][1] - coords[j][1]
        return sqrt(dx*dx + dy*dy)
    
    # Distance matrix
    D = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            D[i,j] = dist(i, j)
            D[j,i] = D[i,j]
    
    # Customer nodes
    cust_nodes = list(range(1, n_customers + 1))
    station_nodes = list(stations)
    
    # State: (node, time, battery)
    # We'll use a Dijkstra-like approach with state (node, time_bucket, battery_level)
    # But time and battery are continuous. Let's discretize or use exact values.
    # Since we need to be exact, let's use a priority queue with states (cost, node, time, battery, visited_mask)
    # But 2^25 is too large.
    
    # Alternative: Constructive heuristic with local search.
    # Let's do a greedy construction with nearest neighbor + time/battery feasibility, then 2-opt local search.
    
    # First, try to build routes greedily.
    
    def is_customer(node):
        return 1 <= node <= n_customers
    
    def is_station(node):
        return node in station_nodes
    
    # Greedy construction: start at depot, visit nearest feasible customer, etc.
    # Feasibility: time window, battery, capacity.
    
    # Let's try a different approach: since n=25 is small, use a simple insertion heuristic.
    # Start with empty routes, insert customers one by one.
    
    # Actually, let's try a simpler approach: nearest neighbor with backtracking if infeasible.
    
    # Let me implement a basic 2-opt with feasibility checks.
    
    # First, create an initial tour using nearest neighbor.
    
    def nearest_neighbor_tour():
        unvisited = set(cust_nodes)
        tour = []
        current = 0
        time = 0.0
        battery = battery_cap
        load = 0
        
        while unvisited:
            # Find nearest feasible customer
            best_node = None
            best_dist = float('inf')
            for v in unvisited:
                d = D[current, v]
                if d >= best_dist:
                    continue
                # Check battery
                if battery - d < 0:
                    continue
                # Check time
                t_arrive = time + d / speed
                if t_arrive > due[v]:
                    continue
                # Check capacity
                if load + demands[v] > capacity:
                    continue
                best_node = v
                best_dist = d
            
            if best_node is None:
                # No feasible customer, break
                break
            
            # Move to best_node
            d = D[current, best_node]
            time += d / speed
            battery -= d
            time = max(time, ready[best_node])
            time += service[best_node]
            load += demands[best_node]
            tour.append(best_node)
            unvisited.remove(best_node)
            current = best_node
        
        # Return to depot
        d = D[current, 0]
        if battery - d >= 0:
            time += d / speed
        # Check horizon
        if time <= horizon:
            return tour
        
        return tour
    
    # This is too simplistic. Let me implement a proper insertion heuristic with multiple routes.
    
    # Given the complexity, let me use a simple but effective approach:
    # 1. Sort customers by due date
    # 2. Assign each customer to the best feasible route (or create new route)
    # 3. Local search with 2-opt within routes and moves between routes
    
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
            elif is_station(node):
                # Recharge
                energy_needed = battery_cap - battery
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
    
    # Insertion heuristic
    def insertion_heuristic():
        customers_sorted = sorted(cust_nodes, key=lambda v: due[v])
        routes = []
        
        for cust in customers_sorted:
            # Try to insert into existing routes
            best_route_idx = -1
            best_pos = -1
            best_cost = float('inf')
            
            for r_idx, route in enumerate(routes):
                for pos in range(len(route) + 1):
                    new_route = route[:pos] + [cust] + route[pos:]
                    feasible, cost = check_route(new_route)
                    if feasible and cost < best_cost:
                        best_cost = cost
                        best_route_idx = r_idx
                        best_pos = pos
            
            if best_route_idx >= 0:
                routes[best_route_idx] = routes[best_route_idx][:best_pos] + [cust] + routes[best_route_idx][best_pos:]
            else:
                # Create new route
                new_route = [cust]
                feasible, cost = check_route(new_route)
                if feasible:
                    routes.append(new_route)
                else:
                    # Should not happen with single customer
                    routes.append(new_route)
        
        return routes
    
    # 2-opt local search within routes
    def two_opt_improve(routes):
        improved = True
        while improved:
            improved = False
            for r_idx in range(len(routes)):
                route = routes[r_idx]
                n = len(route)
                if n < 2:
                    continue
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        # Reverse segment i..j
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        feasible, cost = check_route(new_route)
                        if feasible:
                            old_cost = check_route(route)[1]
                            if cost < old_cost - 1e-9:
                                routes[r_idx] = new_route
                                improved = True
        return routes
    
    # Run insertion + 2-opt
    routes = insertion_heuristic()
    routes = two_opt_improve(routes)
    
    # Ensure all customers are served
    served = set()
    for route in routes:
        for node in route:
            if is_customer(node):
                served.add(node)
    
    missing = set(cust_nodes) - served
    if missing:
        # Add missing customers as single-route entries
        for cust in missing:
            routes.append([cust])
    
    return routes
# EVOLVE-BLOCK-END
