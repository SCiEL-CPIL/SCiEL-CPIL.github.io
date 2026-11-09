import numpy as np
import time

_start_time = None

def _get_time_remaining():
    """Return seconds remaining before the 150s limit (with safety margin)."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    return max(0.0, 145.0 - elapsed)

def _compute_mst_leaves(nodes, start_node, dist_matrix):
    """
    Computes the MST of the graph containing start_node and nodes.
    Returns a list of leaf nodes (excluding start_node).
    """
    n = len(nodes)
    if n == 0:
        return []
    
    # Map node ids to indices 0..n for local matrix
    # We need a distance matrix for the subgraph {start_node} U nodes
    # Size: (n+1) x (n+1)
    
    # For efficiency with small N, we can just use Prim's on the indices
    # Let's use 0 as index for start_node, and 1..n for nodes
    
    # Build local distance matrix
    # dist[i, j]
    # i=0 is start_node
    # i>0 corresponds to nodes[i-1]
    
    # We can avoid building the full matrix if we just access dist_matrix directly
    # But Prim's needs fast access.
    
    # Let's just use the global indices in Prim's
    
    visited = [False] * (n + 1)
    visited[0] = True # start_node is in MST
    
    # min_dist[i] is the min distance from any visited node to node i
    # node i in local indices: 0 is start, 1..n are unvisited
    # We need to map local index to global node id
    # local 0 -> start_node
    # local i -> nodes[i-1]
    
    min_dist = [np.inf] * (n + 1)
    # Initialize min_dist for neighbors of start_node
    for i in range(1, n + 1):
        global_node = nodes[i-1]
        min_dist[i] = dist_matrix[start_node, global_node]
    
    # Prim's Algorithm
    for _ in range(n):
        # Find unvisited node with min_dist
        u = -1
        min_val = np.inf
        for i in range(1, n + 1):
            if not visited[i] and min_dist[i] < min_val:
                min_val = min_dist[i]
                u = i
        
        if u == -1:
            break # Should not happen for connected graph
            
        visited[u] = True
        
        # Update neighbors
        global_u = nodes[u-1]
        for v in range(1, n + 1):
            if not visited[v]:
                global_v = nodes[v-1]
                d = dist_matrix[global_u, global_v]
                if d < min_dist[v]:
                    min_dist[v] = d
    
    # Now we have the MST edges implicitly? No, we need the degrees.
    # To get degrees, we need to know which edges were selected.
    # Let's re-run Prim's and store the parent of each node.
    
    visited = [False] * (n + 1)
    visited[0] = True
    min_dist = [np.inf] * (n + 1)
    parent = [-1] * (n + 1) # parent[i] is the local index of the parent in MST
    
    for i in range(1, n + 1):
        global_node = nodes[i-1]
        min_dist[i] = dist_matrix[start_node, global_node]
        parent[i] = 0
        
    for _ in range(n):
        u = -1
        min_val = np.inf
        for i in range(1, n + 1):
            if not visited[i] and min_dist[i] < min_val:
                min_val = min_dist[i]
                u = i
                
        if u == -1:
            break
            
        visited[u] = True
        
        global_u = nodes[u-1]
        for v in range(1, n + 1):
            if not visited[v]:
                global_v = nodes[v-1]
                d = dist_matrix[global_u, global_v]
                if d < min_dist[v]:
                    min_dist[v] = d
                    parent[v] = u
    
    # Calculate degrees
    degrees = [0] * (n + 1)
    for i in range(1, n + 1):
        degrees[parent[i]] += 1
        degrees[i] += 1
        
    # Leaves are nodes with degree 1
    leaves = []
    for i in range(1, n + 1):
        if degrees[i] == 1:
            leaves.append(nodes[i-1])
            
    return leaves

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    if _get_time_remaining() < 1.0:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    if len(unvisited_nodes) == 0:
        return destination_node
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Try MST Leaf Heuristic
    try:
        leaves = _compute_mst_leaves(unvisited_nodes, current_node, distance_matrix)
        if leaves:
            # Select leaf with min distance to current_node
            leaf_dists = distance_matrix[current_node, leaves]
            best_leaf = leaves[np.argmin(leaf_dists)]
            return int(best_leaf)
        else:
            # Fallback to nearest neighbor
            return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    except Exception:
        # Fallback
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
