import numpy as np
import time
import math
from itertools import combinations

def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # Extract instance data
    coords = np.array(instance["coords"], dtype=np.float64)
    n_customers = instance["n_customers"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    customers = list(range(1, n_customers + 1))
    n_nodes = len(coords)

    # Pre-compute distance matrix
    dist = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            d = math.sqrt(dx * dx + dy * dy)
            dist[i, j] = d
            dist[j, i] = d

    def simulate_route(route):
        """Simulate a route and return (is_valid, total_distance)."""
        if not route:
            return True, 0.0

        cur_node = 0
        cur_time = 0.0
        cur_bat = battery
        total_dist = 0.0
        load = 0.0

        for node in route:
            d = dist[cur_node, node]
            cur_time += d / speed
            cur_bat -= d
            total_dist += d

            if cur_bat < -1e-9:
                return False, total_dist

            if node in stations:
                energy = battery - cur_bat
                cur_time += energy * recharge_rate
                cur_bat = battery
            else:
                load += demands[node]
                if load > capacity + 1e-9:
                    return False, total_dist
                cur_time = max(cur_time, ready[node])
                if cur_time > due[node] + 1e-9:
                    return False, total_dist
                cur_time += service[node]

            cur_node = node

        d = dist[cur_node, 0]
        cur_time += d / speed
        cur_bat -= d
        total_dist += d

        if cur_bat < -1e-9:
            return False, total_dist
        if cur_time > horizon + 1e-9:
            return False, total_dist

        return True, total_dist

    def route_distance(route):
        """Compute total distance of a route including depot legs."""
        if not route:
            return 0.0
        total = 0.0
        cur_node = 0
        for node in route:
            total += dist[cur_node, node]
            cur_node = node
        total += dist[cur_node, 0]
        return total

    def generate_columns(customers_list, time_limit):
        """
        Column generator: generates a set of feasible routes covering subsets of customers.
        Uses k-Tree heuristic (nearest neighbor with look-ahead) to generate diverse, high-quality routes.
        """
        columns = []
        seen_routes = set()
        start_gen = time.time()
        
        # Generate routes using different starting points and heuristics
        for start_c in customers_list:
            if time.time() > time_limit:
                break
            
            # Strategy 1: Nearest Neighbor
            route = []
            unvisited = set(customers_list)
            cur_node = 0
            cur_load = 0
            cur_bat = battery
            cur_time = 0.0
            
            while unvisited:
                best_node = -1
                best_cost = float('inf')
                
                for c in unvisited:
                    if cur_load + demands[c] > capacity:
                        continue
                    
                    d = dist[cur_node, c]
                    new_time = cur_time + d / speed
                    new_bat = cur_bat - d
                    
                    if new_bat < -1e-9:
                        continue
                    
                    # Check time windows
                    if c not in stations:
                        if new_time < ready[c]:
                            new_time = ready[c]
                        if new_time > due[c] + 1e-9:
                            continue
                        new_time += service[c]
                    
                    # Cost: distance + time penalty
                    cost = d + 0.1 * (new_time - cur_time)
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_node = c
                
                if best_node == -1:
                    break
                
                route.append(best_node)
                cur_load += demands[best_node]
                cur_bat -= dist[cur_node, best_node]
                cur_time += dist[cur_node, best_node] / speed
                
                if best_node in stations:
                    energy = battery - cur_bat
                    cur_time += energy * recharge_rate
                    cur_bat = battery
                else:
                    cur_time = max(cur_time, ready[best_node])
                    cur_time += service[best_node]
                
                cur_node = best_node
                unvisited.remove(best_node)
            
            if route:
                is_valid, total_d = simulate_route(route)
                if is_valid:
                    route_tuple = tuple(sorted(route))
                    if route_tuple not in seen_routes:
                        seen_routes.add(route_tuple)
                        columns.append((route, total_d))

        # Strategy 2: Regret-based construction for more diversity
        for start_c in customers_list[:10]: # Limit to save time
            if time.time() > time_limit:
                break
            
            route = []
            unvisited = set(customers_list)
            cur_node = 0
            cur_load = 0
            cur_bat = battery
            cur_time = 0.0
            
            while unvisited:
                # Calculate regret for each customer
                best_costs = {}
                for c in unvisited:
                    if cur_load + demands[c] > capacity:
                        continue
                    d = dist[cur_node, c]
                    new_time = cur_time + d / speed
                    new_bat = cur_bat - d
                    if new_bat < -1e-9:
                        continue
                    if c not in stations:
                        if new_time < ready[c]:
                            new_time = ready[c]
                        if new_time > due[c] + 1e-9:
                            continue
                        new_time += service[c]
                    best_costs[c] = d
                
                if not best_costs:
                    break
                
                # Pick customer with highest regret (difference between 1st and 2nd best insertion cost approximation)
                # Here we approximate by picking the one that minimizes distance but is "urgent"
                sorted_costs = sorted(best_costs.items(), key=lambda x: x[1])
                if len(sorted_costs) > 1:
                    regret = sorted_costs[1][1] - sorted_costs[0][1]
                    # Simple heuristic: if regret is high, prioritize that customer? 
                    # For simplicity, just pick the nearest feasible one but with a slight bias towards urgent ones
                    best_node = sorted_costs[0][0]
                else:
                    best_node = sorted_costs[0][0]
                
                route.append(best_node)
                cur_load += demands[best_node]
                cur_bat -= dist[cur_node, best_node]
                cur_time += dist[cur_node, best_node] / speed
                
                if best_node in stations:
                    energy = battery - cur_bat
                    cur_time += energy * recharge_rate
                    cur_bat = battery
                else:
                    cur_time = max(cur_time, ready[best_node])
                    cur_time += service[best_node]
                
                cur_node = best_node
                unvisited.remove(best_node)
            
            if route:
                is_valid, total_d = simulate_route(route)
                if is_valid:
                    route_tuple = tuple(sorted(route))
                    if route_tuple not in seen_routes:
                        seen_routes.add(route_tuple)
                        columns.append((route, total_d))

        return columns

    def solve_set_partitioning(columns, customers, time_limit):
        """
        Solve the set partitioning problem using Branch and Bound.
        Minimize sum of route costs subject to each customer being covered exactly once.
        """
        n_cols = len(columns)
        if n_cols == 0:
            return None, float('inf')
        
        # Pre-compute customer masks for each column
        col_masks = []
        for route, cost in columns:
            mask = 0
            for c in route:
                mask |= (1 << (c - 1)) # c is 1..25, so shift by c-1
            col_masks.append(mask)
        
        full_mask = (1 << n_customers) - 1
        
        best_solution = None
        best_cost = float('inf')
        
        def branch_and_bound(node_idx, current_mask, current_cost, selected_cols):
            nonlocal best_solution, best_cost
            
            if time.time() > time_limit:
                return
            
            # Check if we've covered all customers
            if current_mask == full_mask:
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_solution = list(selected_cols)
                return
            
            # Check if any remaining customers can be covered
            remaining_mask = full_mask & ~current_mask
            if remaining_mask == 0:
                return
            
            # Lower bound: current cost + minimum cost to cover remaining customers
            # Greedy lower bound: for each remaining customer, find min cost column covering it
            lb = current_cost
            temp_mask = remaining_mask
            while temp_mask:
                # Find lowest bit
                lsb = temp_mask & -temp_mask
                c_idx = (lsb.bit_length() - 1) + 1 # customer id
                
                min_cost = float('inf')
                for i in range(node_idx, n_cols):
                    if col_masks[i] & lsb:
                        if columns[i][1] < min_cost:
                            min_cost = columns[i][1]
                
                if min_cost == float('inf'):
                    return # Cannot cover this customer with remaining columns
                
                lb += min_cost
                temp_mask ^= lsb
                
            if lb >= best_cost:
                return
            
            # Branch on next column
            for i in range(node_idx, n_cols):
                new_mask = current_mask | col_masks[i]
                
                # Check for conflict (customer covered twice)
                if new_mask != current_mask and (new_mask & ~current_mask) != (col_masks[i] & ~current_mask):
                    # This check is redundant if we just OR, but we need to ensure no overlap with already selected
                    # Actually, set partitioning requires disjointness.
                    # If (col_masks[i] & current_mask) != 0, there is an overlap.
                    pass
                
                if col_masks[i] & current_mask:
                    continue # Overlap, invalid for partitioning
                
                new_cost = current_cost + columns[i][1]
                
                if new_cost < best_cost:
                    branch_and_bound(i + 1, new_mask, new_cost, selected_cols + [i])
        
        branch_and_bound(0, 0, 0, [])
        
        if best_solution:
            routes = [columns[i][0] for i in best_solution]
            return routes, best_cost
        return None, float('inf')

    # Main Logic
    # 1. Generate Columns
    col_time_limit = 20.0
    columns = generate_columns(customers, col_time_limit)
    
    # 2. Solve Set Partitioning
    sp_time_limit = deadline - 5.0 - time.time()
    routes, total_dist = solve_set_partitioning(columns, customers, sp_time_limit)
    
    # Fallback if B&B fails or times out without solution
    if routes is None:
        # Use a greedy covering heuristic as fallback
        routes = []
        unvisited = set(customers)
        
        # Sort columns by cost per customer (efficiency)
        sorted_cols = sorted(columns, key=lambda x: x[1] / max(1, len(x[0])))
        
        for route, cost in sorted_cols:
            if time.time() > deadline - 2.0:
                break
            if not unvisited:
                break
            
            # Check if route only contains unvisited customers
            valid_route = [c for c in route if c in unvisited]
            if valid_route:
                # Re-simulate to ensure validity (should be valid by construction)
                is_valid, _ = simulate_route(valid_route)
                if is_valid:
                    routes.append(valid_route)
                    for c in valid_route:
                        unvisited.remove(c)
        
        # If still unvisited, create singleton routes
        for c in unvisited:
            routes.append([c])
    
    return routes
