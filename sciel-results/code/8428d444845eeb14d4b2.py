import numpy as np
import sys

# Global cache for MST tours
_mst_tour_cache = {}

def _get_mst_tour(distance_matrix):
    # Create a simple hash for the matrix to use as a cache key
    # Using a tuple of the first few elements and shape is a cheap proxy
    # Or just use id() if we assume unique objects per instance
    key = id(distance_matrix)
    
    if key in _mst_tour_cache:
        return _mst_tour_cache[key]

    n = distance_matrix.shape[0]
    # Prim's Algorithm for MST
    in_mst = [False] * n
    min_dist = [np.inf] * n
    min_dist[0] = 0
    parent = [-1] * n
    
    for _ in range(n):
        # Find min dist node not in MST
        u = -1
        best = np.inf
        for i in range(n):
            if not in_mst[i] and min_dist[i] < best:
                best = min_dist[i]
                u = i
        
        if u == -1:
            break # Should not happen for connected graph
            
        in_mst[u] = True
        
        # Update neighbors
        for v in range(n):
            if not in_mst[v] and distance_matrix[u, v] < min_dist[v]:
                min_dist[v] = distance_matrix[u, v]
                parent[v] = u

    # Build adjacency list
    adj = [[] for _ in range(n)]
    for i in range(1, n):
        p = parent[i]
        adj[p].append(i)
        adj[i].append(p)
        
    # DFS to get tour order (Pre-order)
    tour = []
    visited = [False] * n
    
    def dfs(node):
        visited[node] = True
        tour.append(node)
        for neighbor in adj[node]:
            if not visited[neighbor]:
                dfs(neighbor)
                
    dfs(0)
    
    _mst_tour_cache[key] = tour
    return tour

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    tour = _get_mst_tour(distance_matrix)
    
    # Find current node's index in the tour
    # Since n=50, linear search is fast
    try:
        idx = tour.index(current_node)
    except ValueError:
        # Fallback to nearest neighbor if something goes wrong
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
        
    # Next node in the circular tour
    next_idx = (idx + 1) % len(tour)
    next_node = tour[next_idx]
    
    # Safety check: ensure next_node is in unvisited_nodes
    # This should always be true if we follow the tour correctly
    if next_node not in unvisited_nodes:
        # If not, it means we might have wrapped around or there's a state mismatch
        # Fallback to nearest neighbor among unvisited
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
        
    return int(next_node)
