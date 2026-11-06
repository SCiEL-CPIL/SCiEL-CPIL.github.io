import numpy as np

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

    # Distance from current to all feasible nodes
    dist_current_to_feasible = distance_matrix[current_node, feasible_nodes]
    
    best_score = np.inf
    best_node = int(feasible_nodes[0])
    
    # Pre-convert demands to list for faster access in loops if needed, but numpy array is fine
    # We simulate the rest of the tour using a greedy nearest-neighbor approach
    # for each candidate node to estimate the total route length.
    
    for i in range(M):
        j = int(feasible_nodes[i])
        
        # Cost to reach j from current
        cost = dist_current_to_feasible[i]
        
        # State after visiting j
        curr_loc = j
        rem_cap = rest_capacity - int(demands[j])
        
        # Remaining nodes: all unvisited except j
        # Use a list for efficient removal
        rem_nodes_list = [int(x) for x in unvisited if x != j]
        
        # Greedy simulation of the rest of the route
        while rem_nodes_list:
            # Filter feasible nodes from remaining
            # Using list comprehension for small N is fast
            feasible_rem = [n for n in rem_nodes_list if demands[n] <= rem_cap]
            
            if not feasible_rem:
                break
                
            # Find nearest feasible node
            arr = np.array(feasible_rem)
            dists = distance_matrix[curr_loc, arr]
            min_idx = np.argmin(dists)
            next_node = int(arr[min_idx])
            
            cost += dists[min_idx]
            curr_loc = next_node
            rem_cap -= int(demands[next_node])
            
            # Remove next_node from remaining list
            # List removal is O(N), N<=50, so this is fine
            rem_nodes_list.remove(next_node)
            
        # Return to depot
        cost += distance_matrix[curr_loc, depot]
        
        if cost < best_score:
            best_score = cost
            best_node = j
            
    return best_node
