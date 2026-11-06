import numpy as np
import heapq

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    if len(unvisited_nodes) == 0:
        return depot

    unvisited = np.array(unvisited_nodes)
    unvisited_demands = demands[unvisited]
    
    # Feasible nodes: demand <= rest_capacity
    feasible_mask = unvisited_demands <= rest_capacity
    
    if not np.any(feasible_mask):
        return depot

    feasible_indices = np.where(feasible_mask)[0]
    feasible_nodes = unvisited[feasible_indices]
    M = len(feasible_nodes)

    if M == 1:
        return int(feasible_nodes[0])

    best_score = np.inf
    best_node = int(feasible_nodes[0])
    
    # Precompute distances from current to feasible nodes
    dist_current_to_feasible = distance_matrix[current_node, feasible_nodes]
    
    # Convert demands to list for faster access
    demands_list = demands.tolist()
    dist_mat = distance_matrix
    
    for i in range(M):
        j = int(feasible_nodes[i])
        
        # Cost to reach j from current
        cost_to_j = dist_current_to_feasible[i]
        
        # State after visiting j
        rem_cap = rest_capacity - int(demands_list[j])
        
        # Remaining nodes: all unvisited except j
        rem_nodes = [int(x) for x in unvisited if x != j]
        
        # Calculate Dijkstra cost from j to depot on remaining nodes
        # State: (cost, node, cap)
        
        if not rem_nodes:
            dijkstra_cost = dist_mat[j, depot]
            nearest_dist = 0.0
        else:
            # Initialize distances
            start_cap = rem_cap
            dists = {}
            dists[(j, start_cap)] = 0.0
            heap = [(0.0, j, start_cap)]
            
            min_depot_cost = np.inf
            
            while heap:
                c, u, cap_u = heapq.heappop(heap)
                
                # If we reached depot, update min and break (Dijkstra property)
                if u == depot:
                    min_depot_cost = c
                    break
                
                # Check if this state was already processed with lower cost
                if c > dists.get((u, cap_u), np.inf):
                    continue

                # Explore neighbors
                # 1. Go to depot
                new_cost = c + dist_mat[u, depot]
                if new_cost < min_depot_cost:
                    # We don't need to store depot states in dists if we just want min cost
                    # But we might reach depot via different caps?
                    # We break on first pop, so we don't need to push depot?
                    # Actually, we can just check if u==depot at pop time.
                    # So we don't need to push depot into heap if we handle it at pop?
                    # No, we must push it to be popped.
                    # But if we push depot, we need to store it in dists?
                    # If we push (new_cost, depot, cap_u), and later pop it,
                    # we check u==depot.
                    # We should update min_depot_cost when we *push*? No, when we *pop*.
                    # So we push depot states.
                    if (depot, cap_u) not in dists or new_cost < dists[(depot, cap_u)]:
                        dists[(depot, cap_u)] = new_cost
                        heapq.heappush(heap, (new_cost, depot, cap_u))

                # 2. Go to remaining customers
                for k in rem_nodes:
                    if demands_list[k] <= cap_u:
                        new_cap = cap_u - int(demands_list[k])
                        nc = c + dist_mat[u, k]
                        if (k, new_cap) not in dists or nc < dists[(k, new_cap)]:
                            dists[(k, new_cap)] = nc
                            heapq.heappush(heap, (nc, k, new_cap))
            
            if np.isinf(min_depot_cost):
                # Should not happen if feasible
                dijkstra_cost = np.inf
            else:
                dijkstra_cost = min_depot_cost

            # Calculate nearest feasible node from j in rem_nodes
            nearest_dist = np.inf
            for k in rem_nodes:
                if demands_list[k] <= rem_cap:
                    d = dist_mat[j, k]
                    if d < nearest_dist:
                        nearest_dist = d
            
            if not np.isfinite(nearest_dist):
                nearest_dist = 0.0

        total_score = cost_to_j + dijkstra_cost + nearest_dist
        
        if total_score < best_score:
            best_score = total_score
            best_node = j

    return best_node
