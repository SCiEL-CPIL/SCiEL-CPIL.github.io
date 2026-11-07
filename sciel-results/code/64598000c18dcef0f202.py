import numpy as np
import time
import heapq

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes a heuristic matrix for ACO.
    Strategy: Data-driven heuristic construction using Oracle Solutions.
    1. Generate high-quality VRP solutions using a fast meta-heuristic 
       (Savings Algorithm Construction + Intra-route 2-Opt).
    2. Aggregate edge frequencies from these solutions.
    3. Construct the heuristic matrix based on these frequencies, 
       biasing the ACO towards edges that are part of known good solutions.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin for 300s limit

    n = len(demands)
    demands = np.array(demands, dtype=float)
    
    # Helper: Fast VRP solver using Savings Algorithm + 2-Opt
    def solve_vrp_oracle(num_starts=100):
        best_routes = []
        best_cost = float('inf')
        
        # Pre-compute savings: s(i, j) = d(i, 0) + d(0, j) - d(i, j)
        # d(i, 0) is distance from i to depot.
        # We want to maximize savings.
        
        for start_idx in range(num_starts):
            if time.time() > deadline - 10.0:
                break
            
            # Randomize the savings order slightly to get diverse solutions
            # Standard Savings Algorithm sorts by savings descending.
            # We can shuffle ties or add small noise to break symmetry.
            
            # 1. Initial state: Each customer is its own route: [0, i, 0]
            # We represent routes as a list of lists.
            # To make it fast, we might use a union-find or just lists.
            # With n=50, lists are fine.
            
            routes = []
            for i in range(1, n):
                routes.append([0, i, 0])
            
            # Calculate savings for all pairs
            # s(i, j) = d(i, 0) + d(0, j) - d(i, j)
            # Only consider pairs where capacity allows merging?
            # Actually, standard savings doesn't check capacity during merge,
            # but we must ensure the resulting route fits in capacity.
            
            savings_list = []
            for i in range(1, n):
                for j in range(i + 1, n):
                    s = distance_matrix[i, 0] + distance_matrix[0, j] - distance_matrix[i, j]
                    # Check if they can be in the same route
                    # They can be in the same route if demand[i] + demand[j] <= capacity
                    # But in the final route, they might be with others.
                    # For the initial merge, we just check if the two nodes
                    # can be adjacent in a route that fits.
                    # A safe check: if demand[i] + demand[j] <= capacity, 
                    # then there exists *some* route containing them.
                    # The Savings algorithm merges routes.
                    # Let's pre-filter pairs that are definitely impossible.
                    if demands[i] + demands[j] <= capacity:
                        savings_list.append((s, i, j))
            
            # Sort by savings descending
            # To add diversity, we can sort with a slight random perturbation
            # or just pick a random subset of top-k?
            # Let's sort normally, but maybe randomize the initial route order?
            # Actually, randomizing the tie-breaking or the order of consideration 
            # when savings are close is good.
            # For simplicity and speed, let's just sort.
            # To get different solutions, we can randomize the order of customers 
            # in the initial "routes" list? No, that doesn't change savings.
            # We can add noise to savings: s' = s + noise
            noise = np.random.uniform(0, 0.01 * np.max([abs(s[0]) for s in savings_list] if savings_list else [1]))
            # Applying noise to all is expensive. 
            # Alternative: Just run standard savings. 
            # To get diversity, we can run it once, then modify the resulting 
            # routes with 2-Opt.
            
            savings_list.sort(key=lambda x: x[0], reverse=True)
            
            # Map node to its current route index
            # route_idx[i] = index of the route containing node i
            # We also need to know the position of i in that route to merge correctly.
            # Since routes are [0, ..., 0], the internal nodes are the customers.
            # When merging Route A and Route B, we remove one 0 and connect the end of A to start of B.
            # Specifically, if A ends with i (i.e., [..., i, 0]) and B starts with j (i.e., [0, j, ...]),
            # we merge to [..., i, j, ...].
            # So we need to track the "open" ends.
            
            # Data structure:
            # routes: list of lists of customer nodes (excluding depot)
            # But we need to know the order.
            # Let's store routes as lists of customers.
            # The actual path is 0 -> r[0] -> r[1] -> ... -> r[-1] -> 0.
            
            routes = [[i] for i in range(1, n)]
            # parent[i] = index of route containing i
            # end[i] = is i the last node in its route? 
            # Actually, when we merge, we connect the end of one route to the start of another.
            # So we need to know which nodes are "ends" (connected to depot) and which are internal.
            # In the initial state, all nodes are both start and end.
            # When we merge Route A (ending in i) and Route B (starting with j), 
            # i and j become internal.
            
            # Let's use a simpler representation for n=50:
            # Just keep the list of routes as lists of customers.
            # To find if i and j can be merged, we check if i is the last element of its route 
            # and j is the first element of its route (or vice versa).
            # Savings s(i, j) implies we want to connect i and j.
            # This requires i to be an end of its route and j to be an end of its route.
            
            # Efficient check:
            # is_end[i] = True if i is at the start or end of its route.
            # Actually, in the standard algorithm, we only merge if i is the last node of route A 
            # and j is the first node of route B (or vice versa).
            
            # Let's maintain:
            # route_of_node[i] = index of the route
            # routes: list of lists
            
            route_of_node = {}
            for i in range(1, n):
                route_of_node[i] = i - 1 # Initial route index
            
            # is_start[i] and is_end[i]
            is_start = {i: True for i in range(1, n)}
            is_end = {i: True for i in range(1, n)}
            
            # Process savings
            merged_count = 0
            for s, i, j in savings_list:
                if time.time() > deadline - 5.0:
                    break
                
                # Check if i and j are in different routes
                if route_of_node[i] == route_of_node[j]:
                    continue
                
                # Check if i and j are "ends" of their respective routes
                # Case 1: i is end of Route A, j is start of Route B
                # Case 2: j is end of Route B, i is start of Route A
                
                r_i = route_of_node[i]
                r_j = route_of_node[j]
                
                # Check capacity
                load_i = sum(demands[k] for k in routes[r_i])
                load_j = sum(demands[k] for k in routes[r_j])
                if load_i + load_j > capacity:
                    continue
                
                # Check end/start status
                # We need to know if i is at the end of routes[r_i] and j is at the start of routes[r_j]
                # OR if j is at the end of routes[r_j] and i is at the start of routes[r_i]
                
                # To make this O(1), we can store the first and last element of each route.
                # But routes are lists. 
                # For n=50, checking routes[r_i][-1] == i is O(1).
                
                if routes[r_i][-1] == i and routes[r_j][0] == j:
                    # Merge: Remove i from end of r_i, remove j from start of r_j
                    # New route: routes[r_i] + routes[r_j]
                    # Wait, if we remove i and j, we connect i and j.
                    # The new list is routes[r_i] (which ends in i) + routes[r_j] (which starts with j).
                    # Actually, we keep i and j in the list, they just become adjacent.
                    # So new_route = routes[r_i] + routes[r_j]
                    # But we must remove the depot? No, we stored customers only.
                    # So new_route = routes[r_i] + routes[r_j]
                    # The connection i->j is now internal.
                    
                    new_route = routes[r_i] + routes[r_j]
                    
                    # Update is_start/is_end
                    # i is no longer an end
                    is_end[i] = False
                    # j is no longer a start
                    is_start[j] = False
                    
                    # Update route_of_node for all nodes in r_j
                    for node in routes[r_j]:
                        route_of_node[node] = r_i
                    
                    # Replace route r_i with new_route, remove r_j
                    routes[r_i] = new_route
                    routes[r_j] = [] # Mark as empty
                    merged_count += 1
                    
                elif routes[r_j][-1] == j and routes[r_i][0] == i:
                    # Merge: j is end of r_j, i is start of r_i
                    new_route = routes[r_j] + routes[r_i]
                    
                    is_end[j] = False
                    is_start[i] = False
                    
                    for node in routes[r_i]:
                        route_of_node[node] = r_j
                    
                    routes[r_j] = new_route
                    routes[r_i] = []
                    merged_count += 1
            
            # Calculate cost of these routes
            cost = 0
            for route in routes:
                if not route:
                    continue
                # Path: 0 -> route[0] -> ... -> route[-1] -> 0
                prev = 0
                for node in route:
                    cost += distance_matrix[prev, node]
                    prev = node
                cost += distance_matrix[prev, 0]
            
            # Apply Intra-route 2-Opt
            improved_routes = []
            for route in routes:
                if not route:
                    continue
                # 2-Opt on the sequence of customers
                # We fix the depot at start and end.
                # We reverse segments within the customer list.
                improved = True
                max_iter = 10
                while improved and max_iter > 0:
                    improved = False
                    max_iter -= 1
                    for i in range(len(route) - 1):
                        for j in range(i + 1, len(route)):
                            # Reverse route[i:j+1]
                            # Check cost change
                            # Edges involved:
                            # Before: (prev, route[i]), (route[j], next)
                            # After: (prev, route[j]), (route[i], next)
                            # prev is route[i-1] if i>0 else 0
                            # next is route[j+1] if j < len(route)-1 else 0
                            
                            prev_node = route[i-1] if i > 0 else 0
                            next_node = route[j+1] if j < len(route) - 1 else 0
                            
                            old_cost = distance_matrix[prev_node, route[i]] + distance_matrix[route[j], next_node]
                            new_cost = distance_matrix[prev_node, route[j]] + distance_matrix[route[i], next_node]
                            
                            if new_cost < old_cost - 1e-10:
                                route[i:j+1] = reversed(route[i:j+1])
                                improved = True
                                # Note: After reversal, is_start/is_end might change?
                                # No, the ends of the route (first and last customer) 
                                # only change if i=0 or j=len-1.
                                # But for 2-Opt, we usually don't change the connection to depot
                                # unless we reverse the whole route or part of it touching ends.
                                # If i=0, route[0] becomes route[j]. The start node changes.
                                # If j=len-1, route[-1] becomes route[i]. The end node changes.
                                # This is fine for cost calculation.
                                # We don't need to update is_start/is_end for 2-Opt 
                                # because we are done with merging.
            
            improved_routes.append(route)
            
            # Recalculate cost after 2-Opt
            final_cost = 0
            for route in improved_routes:
                if not route:
                    continue
                prev = 0
                for node in route:
                    final_cost += distance_matrix[prev, node]
                    prev = node
                final_cost += distance_matrix[prev, 0]
            
            if final_cost < best_cost:
                best_cost = final_cost
                best_routes = improved_routes
                
            # If we have enough time, we could try more starts
            # But 100 starts of Savings+2Opt should be fast enough for n=50.
            
        return best_routes, best_cost

    # Run the oracle solver
    oracle_routes, oracle_cost = solve_vrp_oracle(num_starts=50)
    
    # If oracle failed or took too long, fallback to 1/distance
    if not oracle_routes:
        safe_dist = distance_matrix + 1e-9
        result = 1.0 / safe_dist
        result[~np.isfinite(result)] = 1e-9
        result[result < 1e-9] = 1e-9
        return result

    # Construct heuristic matrix from oracle routes
    # Count edge frequencies
    edge_freq = np.zeros((n, n))
    
    for route in oracle_routes:
        if not route:
            continue
        prev = 0
        for node in route:
            edge_freq[prev, node] += 1
            edge_freq[node, prev] += 1 # Symmetric? 
            # The problem uses distance_matrix which is likely symmetric, 
            # but ACO might treat directed edges. 
            # The heuristic eta[i, j] is used for transition i->j.
            # If the route goes i->j, then eta[i, j] should be high.
            # Since VRP routes are undirected in terms of cost, 
            # but the sequence is directed, we should count directed edges.
            # However, 2-Opt can reverse segments, so both directions might appear.
            # Let's count directed edges as they appear in the route.
            prev = node
        edge_freq[prev, 0] += 1
        edge_freq[0, prev] += 1

    # Normalize and combine with distance
    # High frequency -> High attractiveness.
    # Also, shorter distance -> Higher attractiveness.
    # eta[i, j] = (1 + edge_freq[i, j]) / distance[i, j]
    # Or eta[i, j] = (1 + edge_freq[i, j]) * (1 / distance[i, j])
    
    safe_dist = distance_matrix + 1e-9
    # Boost edges that are in the oracle solution
    # If an edge is in the solution, it gets a big boost.
    # If not, it falls back to 1/dist.
    
    # Let's try: eta[i, j] = (1 + edge_freq[i, j] * 10) / safe_dist[i, j]
    # The factor 10 makes oracle edges significantly more attractive.
    
    result = (1.0 + edge_freq * 10.0) / safe_dist
    
    # Safety checks
    result[~np.isfinite(result)] = 1e-9
    result[result < 1e-9] = 1e-9
    
    return result
