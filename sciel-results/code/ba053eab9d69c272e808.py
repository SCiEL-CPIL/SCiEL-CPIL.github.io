# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

# Global deadline guard
_START_TIME = time.time()
_TIMEOUT_LIMIT = 140.0 # Leave 10s margin for 150s limit

def _is_timeout():
    return (time.time() - _START_TIME) > _TIMEOUT_LIMIT

def _mst_weight(nodes, distance_matrix):
    """
    Calculate the weight of the Minimum Spanning Tree for the given set of nodes.
    Uses Prim's algorithm with a simple loop, which is sufficient for small N.
    """
    n = len(nodes)
    if n <= 1:
        return 0.0
    
    # Map node indices to 0..n-1 for local matrix access if needed, 
    # but we can just use the global distance_matrix with the node indices.
    
    # Prim's algorithm
    in_mst = [False] * n
    in_mst[0] = True
    total_weight = 0.0
    
    for _ in range(n - 1):
        min_dist = float('inf')
        min_idx = -1
        u_node = nodes[0] # Last added node? No, Prim's keeps track of the frontier.
        
        # We need to track the last added node to expand from, or just find the min edge crossing the cut.
        # Standard Prim's: maintain a priority queue or just scan.
        # Since N is small, scanning is fine.
        
        # Let's use a standard array-based Prim's.
        # dist[i] is the min distance from node i to the current MST.
        # This function is called many times, so we should be careful with overhead.
        
        pass # The simple loop below is better implemented inside the main function for speed if we inline, 
             # but for clarity and correctness first, let's do a standard Prim's.
    
    # Actually, for N=50, a simpler approach:
    # The MST weight is the sum of the edges in the MST.
    # We can compute it by finding the min edge connecting a node in MST to a node not in MST.
    
    # Let's implement a fast Prim's for a subset of nodes.
    # nodes: array of global indices
    # distance_matrix: global matrix
    
    # local indices 0..n-1 correspond to nodes[0..n-1]
    # dist_to_mst[i] for i in 0..n-1
    dist_to_mst = [float('inf')] * n
    dist_to_mst[0] = 0.0
    visited = [False] * n
    
    for _ in range(n):
        # Find unvisited node with min dist_to_mst
        min_val = float('inf')
        u = -1
        for i in range(n):
            if not visited[i] and dist_to_mst[i] < min_val:
                min_val = dist_to_mst[i]
                u = i
        
        if u == -1:
            break # Disconnected? Should not happen in complete graph.
            
        visited[u] = True
        total_weight += min_val
        
        # Update neighbors
        u_global = nodes[u]
        for v in range(n):
            if not visited[v]:
                v_global = nodes[v]
                d = distance_matrix[u_global, v_global]
                if d < dist_to_mst[v]:
                    dist_to_mst[v] = d
                    
    return total_weight

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in a TSP tour using an MST-aware heuristic.
    """
    if _is_timeout():
        # Fallback to nearest neighbor if we are running out of time
        if len(unvisited_nodes) == 0:
            return int(current_node)
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])

    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(current_node)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    if n_unvisited == 2:
        n1, n2 = unvisited_nodes
        # Cost if we go to n1 first: current -> n1 -> n2 -> destination
        cost1 = distance_matrix[current_node, n1] + distance_matrix[n1, n2] + distance_matrix[n2, destination_node]
        # Cost if we go to n2 first: current -> n2 -> n1 -> destination
        cost2 = distance_matrix[current_node, n2] + distance_matrix[n2, n1] + distance_matrix[n1, destination_node]
        return int(n1 if cost1 <= cost2 else n2)

    # For n_unvisited > 2, use MST-aware heuristic
    # Score(u) = d(current, u) + MST(U \ {u})
    
    best_score = float('inf')
    best_node = int(unvisited_nodes[0])
    
    # Pre-compute distances from current to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Iterate over each candidate
    for i, u in enumerate(unvisited_nodes):
        # Immediate cost
        immediate_cost = dist_from_current[i]
        
        # Structural cost: MST of remaining nodes
        remaining_mask = np.ones(n_unvisited, dtype=bool)
        remaining_mask[i] = False
        remaining_nodes = unvisited_nodes[remaining_mask]
        
        # Calculate MST weight of remaining_nodes
        # If remaining is empty, MST weight is 0
        if len(remaining_nodes) == 0:
            mst_w = 0.0
        else:
            mst_w = _mst_weight(remaining_nodes, distance_matrix)
            
        total_score = immediate_cost + mst_w
        
        if total_score < best_score:
            best_score = total_score
            best_node = int(u)
            
    return best_node

# EVOLVE-BLOCK-END
