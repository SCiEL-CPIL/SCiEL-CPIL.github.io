import numpy as np
import time

def plan(instance):
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin before 120s limit

    # 1. Parse and Precompute
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
    
    # Vectorized distance matrix calculation
    coords_diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(coords_diff**2, axis=2))
    
    # Precompute station boolean mask for quick checks
    is_station = np.zeros(n_nodes, dtype=bool)
    for s in stations:
        is_station[s] = True

    # 2. Heuristic Initialization (Savings Algorithm / Clarke-Wright)
    # This is much faster and often yields better initial clusters than simple nearest neighbor insertion,
    # especially for spatial clustering which reduces total distance.
    
    # Start with individual routes for each customer
    # We represent routes as a list of lists of node IDs (excluding depot)
    # But for efficient savings, it's easier to work with a permutation or adjacency.
    # Let's stick to list of lists for clarity and ease of 2-opt implementation later.
    
    # Initial state: Each customer is a separate route [c]
    # To apply savings, we need to merge routes.
    # A route is a sequence of customers.
    # Distance of a route [c1, c2, ..., ck] is d(0, c1) + d(c1, c2) + ... + d(ck, 0).
    # Savings of merging [A] and [B] where A ends with a and B starts with b:
    # Savings = d(0, a) + d(b, 0) - d(a, b)
    # We must ensure capacity, battery, and time windows.
    
    # For 25 customers, O(N^3) or O(N^2 log N) is fine.
    # Let's implement a standard Savings heuristic.
    
    # 1. Create initial routes
    routes = [[c] for c in range(1, n_customers + 1)]
    # Map customer -> route index
    cust_route_idx = {c: i for i, route in enumerate(routes) for c in route}
    
    # Calculate savings for all pairs of customers (i, j)
    # Savings[i][j] = dist[0][i] + dist[0][j] - dist[i][j]
    savings = np.zeros((n_nodes, n_nodes))
    for i in range(1, n_customers + 1):
        for j in range(i + 1, n_customers + 1):
            savings[i][j] = dist[0][i] + dist[0][j] - dist[i][j]
            savings[j][i] = savings[i][j]
            
    # Sort savings in descending order
    # We only need to check pairs where savings > 0
    pairs = []
    for i in range(1, n_customers + 1):
        for j in range(i + 1, n_customers + 1):
            if savings[i][j] > 0:
                pairs.append((savings[i][j], i, j))
                
    pairs.sort(reverse=True)
    
    # Helper to check if two routes can be merged
    def can_merge(route1, route2, c1, c2):
        """
        Try to merge route1 and route2 by connecting c1 (end of route1 or start?) 
        Actually, in savings, we connect the end of one route to the start of another.
        But routes are undirected in terms of "which side" we connect until we fix orientation.
        Usually, we fix the orientation of the smaller route or check both.
        
        Let's define merge(r1, r2, end_idx_r1, start_idx_r2).
        But in the savings list, (i, j) implies we connect i and j.
        i must be the last node of its route, and j must be the first node of its route (or vice versa).
        
        Let's find the positions of i and j in their respective routes.
        """
        r1_idx = cust_route_idx.get(c1)
        r2_idx = cust_route_idx.get(c2)
        if r1_idx is None or r2_idx is None:
            return None, None, False
            
        if r1_idx == r2_idx:
            return None, None, False # Already in same route
            
        r1 = routes[r1_idx]
        r2 = routes[r2_idx]
        
        # Check if c1 is at the end of r1 and c2 is at the start of r2
        # OR c1 is at start of r1 and c2 is at end of r2
        # OR c1 is at end of r2 and c2 is at start of r1
        # OR c1 is at start of r2 and c2 is at end of r1
        
        # We need to check all 4 combinations of orientation for the two routes.
        # Actually, for a specific pair (i, j) from savings, we know we want to link i and j.
        # So i must be at an end of its route, and j must be at an end of its route.
        
        # Case 1: i is end of r1, j is start of r2
        if r1[-1] == c1 and r2[0] == c2:
             new_route = r1 + r2
             if is_feasible(new_route, demands, ready, due, service, dist, battery_max, speed, recharge_rate, horizon, capacity, is_station, n_customers):
                 return r1_idx, r2_idx, True
             
        # Case 2: i is start of r1, j is end of r2
        # This would mean reversing r1: r1[::-1] + r2
        if r1[0] == c1 and r2[-1] == c2:
             new_route = r1[::-1] + r2
             if is_feasible(new_route, demands, ready, due, service, dist, battery_max, speed, recharge_rate, horizon, capacity, is_station, n_customers):
                 return r1_idx, r2_idx, True
                 
        # Case 3: i is end of r2, j is start of r1
        # This would mean r2 + r1
        if r2[-1] == c1 and r1[0] == c2:
             new_route = r2 + r1
             if is_feasible(new_route, demands, ready, due, service, dist, battery_max, speed, recharge_rate, horizon, capacity, is_station, n_customers):
                 return r2_idx, r1_idx, True
                 
        # Case 4: i is start of r2, j is end of r1
        # This would mean r2[::-1] + r1
        if r2[0] == c1 and r1[-1] == c2:
             new_route = r2[::-1] + r1
             if is_feasible(new_route, demands, ready, due, service, dist, battery_max, speed, recharge_rate, horizon, capacity, is_station, n_customers):
                 return r2_idx, r1_idx, True
                 
        return None, None, False

    # 3. Feasibility Check
    def is_feasible(route, demands, ready, due, service, dist, battery_max, speed, recharge_rate, horizon, capacity, is_station, n_customers):
        """
        Check if a route is feasible.
        Returns True if valid, False otherwise.
        """
        if not route:
            return True
            
        curr_node = 0
        curr_time = 0.0
        curr_battery = battery_max
        total_demand = 0.0
        
        # Check capacity first (quick fail)
        for node in route:
            total_demand += demands[node]
            if total_demand > capacity:
                return False
                
        for node in route:
            d = dist[curr_node][node]
            
            # Check battery
            if curr_battery < d:
                return False
            
            curr_battery -= d
            curr_time += d / speed
            
            if is_station[node]:
                # Charge
                charge_time = (battery_max - curr_battery) * recharge_rate
                curr_time += charge_time
                curr_battery = battery_max
            else:
                # Customer
                arrival_time = curr_time
                service_start = max(arrival_time, ready[node])
                if service_start > due[node]:
                    return False
                
                curr_time = service_start + service[node]
                
            curr_node = node
            
        # Return to depot
        d = dist[curr_node][0]
        
        if curr_battery < d:
            return False
        
        curr_battery -= d
        curr_time += d / speed
        
        if curr_time > horizon:
            return False
            
        return True

    # 4. Execute Savings Merge
    for sav, i, j in pairs:
        if time.time() - start_time > DEADLINE - 20.0: # Leave time for local search
            break
            
        idx1, idx2, success = can_merge(routes[idx1] if idx1 is not None else [], routes[idx2] if idx2 is not None else [], i, j)
        
        # The can_merge function above has a slight logic issue: it takes routes but also uses global cust_route_idx.
        # Let's refactor can_merge to be cleaner.
        
    # Refactored Savings Loop
    # We need to update cust_route_idx when we merge.
    
    # Let's restart the savings part with the corrected logic.
    routes = [[c] for c in range(1, n_customers + 1)]
    cust_route_idx = {c: i for i, route in enumerate(routes) for c in route}
    
    for sav, i, j in pairs:
        if time.time() - start_time > DEADLINE - 30.0:
            break
            
        r1_idx = cust_route_idx[i]
        r2_idx = cust_route_idx[j]
        
        if r1_idx == r2_idx:
            continue
            
        r1 = routes[r1_idx]
        r2 = routes[r2_idx]
        
        merged = False
        new_route = None
        new_idx1 = None
        new_idx2 = None
        
        # Try 4 orientations
        # 1. r1 + r2 (i at end of r1, j at start of r2)
        if r1[-1] == i and r2[0] == j:
            candidate = r1 + r2
            if is_feasible(candidate, demands, ready, due, service, dist, battery_max, speed, recharge_rate, horizon, capacity, is_station, n_customers):
                new_route = candidate
                new_idx1 = r1_idx
                new_idx2 = r2_idx
                merged = True
        
        # 2. r1[::-1] + r2 (i at start of r1, j at end of r2)
        if not merged and r1[0] == i and r2[-1] == j:
            candidate = r1[::-1] + r2
            if is_feasible(candidate, demands, ready, due, service, dist, battery_max, speed, recharge_rate, horizon, capacity, is_station, n_customers):
                new_route = candidate
                new_idx1 = r1_idx
                new_idx2 = r2_idx
                merged = True
                
        # 3. r2 + r1 (i at end of r2, j at start of r1)
        if not merged and r2[-1] == i and r1[0] == j:
            candidate = r2 + r1
            if is_feasible(candidate, demands, ready, due, service, dist, battery_max, speed, recharge_rate, horizon, capacity, is_station, n_customers):
                new_route = candidate
                new_idx1 = r2_idx
                new_idx2 = r1_idx
                merged = True
                
        # 4. r2[::-1] + r1 (i at start of r2, j at end of r1)
        if not merged and r2[0] == i and r1[-1] == j:
            candidate = r2[::-1] + r1
            if is_feasible(candidate, demands, ready, due, service, dist, battery_max, speed, recharge_rate, horizon, capacity, is_station, n_customers):
                new_route = candidate
                new_idx1 = r2_idx
                new_idx2 = r1_idx
                merged = True
                
        if merged:
            # Update routes
            routes[new_idx1] = new_route
            # Remove the other route from consideration (mark as empty or remove)
            # We can't easily remove from list without shifting indices.
            # Instead, we set routes[new_idx2] = [] and update cust_route_idx.
            routes[new_idx2] = []
            
            # Update cust_route_idx for all nodes in the new route
            for node in new_route:
                cust_route_idx[node] = new_idx1
                
            # Remove the merged route from the list? 
            # It's safer to keep the list size constant and just ignore empty routes,
            # or filter them out at the end.
            
    # Filter out empty routes
    routes = [r for r in routes if r]
    
    # 5. Local Search: 2-Opt
    # 2-Opt improves the order of nodes within a route.
    
    def two_opt(route):
        if len(route) < 3:
            return route
            
        best_route = route[:]
        best_dist = calculate_route_distance(best_route, dist)
        
        improved = True
        while improved:
            improved = False
            n = len(best_route)
            for i in range(n - 1):
                if time.time() - start_time > DEADLINE:
                    return best_route
                for j in range(i + 1, n):
                    # Reverse segment i..j
                    new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                    
                    # Quick check: does this reverse actually save distance?
                    # Distance change = d(prev, i) + d(j, next) - d(prev, j) - d(i, next)
                    # Where prev is best_route[i-1] or 0, next is best_route[j+1] or 0
                    prev_node = best_route[i-1] if i > 0 else 0
                    next_node = best_route[j+1] if j < n - 1 else 0
                    
                    old_cost = dist[prev_node][best_route[i]] + dist[best_route[j]][next_node]
                    new_cost = dist[prev_node][best_route[j]] + dist[best_route[i]][next_node]
                    
                    if new_cost < old_cost:
                        if is_feasible(new_route, demands, ready, due, service, dist, battery_max, speed, recharge_rate, horizon, capacity, is_station, n_customers):
                            best_route = new_route
                            best_dist = new_cost
                            improved = True
                            break
                if improved:
                    break
                    
        return best_route

    def calculate_route_distance(route, dist):
        if not route:
            return 0.0
        total = 0.0
        curr = 0
        for node in route:
            total += dist[curr][node]
            curr = node
        total += dist[curr][0]
        return total

    # Apply 2-Opt to each route
    for r_idx in range(len(routes)):
        if time.time() - start_time > DEADLINE:
            break
        routes[r_idx] = two_opt(routes[r_idx])
        
    # 6. Inter-route optimization: Relocate and Exchange (Optional, if time permits)
    # Given the small N (25), we can try to move customers between routes to balance load and distance.
    # However, 2-Opt within routes is usually the biggest gain.
    # Let's try a simple Relocate heuristic: Move a customer from one route to another if it reduces total distance.
    
    def try_relocate(routes):
        improved = False
        for r1_idx in range(len(routes)):
            if not routes[r1_idx]: continue
            for r2_idx in range(len(routes)):
                if r1_idx == r2_idx or not routes[r2_idx]: continue
                
                r1 = routes[r1_idx]
                r2 = routes[r2_idx]
                
                # Try moving each customer from r1 to r2
                for c_pos in range(len(r1)):
                    c = r1[c_pos]
                    
                    # Remove c from r1
                    new_r1 = r1[:c_pos] + r1[c_pos+1:]
                    
                    # Try inserting c into all positions in r2
                    for p in range(len(r2) + 1):
                        new_r2 = r2[:p] + [c] + r2[p:]
                        
                        # Check feasibility
                        if not is_feasible(new_r1, demands, ready, due, service, dist, battery_max, speed, recharge_rate, horizon, capacity, is_station, n_customers):
                            continue
                        if not is_feasible(new_r2, demands, ready, due, service, dist, battery_max, speed, recharge_rate, horizon, capacity, is_station, n_customers):
                            continue
                        
                        # Calculate distance change
                        old_total = calculate_route_distance(r1, dist) + calculate_route_distance(r2, dist)
                        new_total = calculate_route_distance(new_r1, dist) + calculate_route_distance(new_r2, dist)
                        
                        if new_total < old_total - 1e-6:
                            routes[r1_idx] = new_r1
                            routes[r2_idx] = new_r2
                            improved = True
                            break
                    if improved:
                        break
            if improved:
                break
        return improved

    # Run Relocate until no improvement or time limit
    while time.time() - start_time < DEADLINE - 5.0:
        if not try_relocate(routes):
            break
            
    # Final 2-Opt pass after relocations
    for r_idx in range(len(routes)):
        if time.time() - start_time > DEADLINE:
            break
        routes[r_idx] = two_opt(routes[r_idx])

    return routes
