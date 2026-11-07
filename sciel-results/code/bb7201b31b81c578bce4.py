"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add(tuple(sorted(e)))
    
    # Build adjacency for base graph
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Check if a set of added edges (represented as a set of frozensets or tuples) is valid
    # We'll work with a list of (u, v) where u < v for added edges
    
    def is_valid(added_edges):
        # Check no duplicates with base
        for u, v in added_edges:
            if (u, v) in base_edges:
                return False
            if (v, u) in base_edges:
                return False
        
        # Build full adjacency
        full_adj = [set() for _ in range(n)]
        for u, v in base_edges:
            full_adj[u].add(v)
            full_adj[v].add(u)
        
        for u, v in added_edges:
            full_adj[u].add(v)
            full_adj[v].add(u)
        
        # Check triangle-free
        for u in range(n):
            neighbors = list(full_adj[u])
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    if neighbors[j] in full_adj[neighbors[i]]:
                        return False
        
        # Check connected
        visited = set()
        queue = deque([0])
        visited.add(0)
        while queue:
            node = queue.popleft()
            for neighbor in full_adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        if len(visited) != n:
            return False
        
        # Check diameter <= 4
        # BFS from each node
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                node = queue.popleft()
                for neighbor in full_adj[node]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[node] + 1
                        queue.append(neighbor)
            for d in dist:
                if d > 4:
                    return False
        
        return True
    
    # Greedy approach: try to add edges that reduce diameter
    # Start with no added edges, check if base graph satisfies constraints
    # The base graph is a path, so diameter is 23, which is > 4
    # We need to add edges to reduce diameter to <= 4
    
    # Strategy: Use a heuristic to find a minimal set of edges
    # We'll try to add edges that connect distant parts of the path
    
    # First, let's find a feasible solution by adding edges strategically
    # A path of length 23 needs to be reduced to diameter 4
    
    # One approach: add edges to create a structure with small diameter
    # For example, connect vertices that are far apart
    
    # Let's try a greedy algorithm:
    # 1. Start with the base graph
    # 2. Find the pair of vertices with maximum distance
    # 3. Add an edge that reduces this distance
    # 4. Repeat until diameter <= 4
    
    def get_diameter(full_adj):
        max_dist = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                node = queue.popleft()
                for neighbor in full_adj[node]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[node] + 1
                        queue.append(neighbor)
            for d in dist:
                if d > max_dist:
                    max_dist = d
        return max_dist
    
    def find_farthest_pair(full_adj):
        max_dist = 0
        farthest_pair = None
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                node = queue.popleft()
                for neighbor in full_adj[node]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[node] + 1
                        queue.append(neighbor)
            for end in range(n):
                if dist[end] > max_dist:
                    max_dist = dist[end]
                    farthest_pair = (start, end)
        return farthest_pair, max_dist
    
    # Try different strategies to find a minimal solution
    
    # Strategy 1: Add edges to create a "hub" structure
    # Choose a few hub vertices and connect them to many others
    
    # Let's try to find a solution with a small number of edges
    # We'll use a simple heuristic: add edges between vertices that are far apart
    
    added_edges = []
    full_adj = [set(adj[i]) for i in range(n)]
    
    # Check if current diameter is already <= 4
    while True:
        farthest_pair, max_dist = find_farthest_pair(full_adj)
        if max_dist <= 4:
            break
        
        # Find a good edge to add
        # Try all possible edges and pick the one that reduces diameter the most
        best_edge = None
        best_diameter = max_dist + 1
        
        for u in range(n):
            for v in range(u+1, n):
                if (u, v) in base_edges or (v, u) in base_edges:
                    continue
                if (u, v) in [(a, b) for a, b in added_edges]:
                    continue
                if (v, u) in [(a, b) for a, b in added_edges]:
                    continue
                
                # Check if adding this edge would create a triangle
                # We need to check if there's a vertex w such that both u-w and v-w are edges
                creates_triangle = False
                for w in full_adj[u]:
                    if w in full_adj[v]:
                        creates_triangle = True
                        break
                if creates_triangle:
                    continue
                
                # Add the edge temporarily
                full_adj[u].add(v)
                full_adj[v].add(u)
                
                # Check diameter
                new_diameter = get_diameter(full_adj)
                
                # Remove the edge
                full_adj[u].remove(v)
                full_adj[v].remove(u)
                
                if new_diameter < best_diameter:
                    best_diameter = new_diameter
                    best_edge = (u, v)
        
        if best_edge is None:
            # No valid edge to add, try a different approach
            break
        
        # Add the best edge
        u, v = best_edge
        added_edges.append((u, v))
        full_adj[u].add(v)
        full_adj[v].add(u)
    
    # Verify the solution
    if is_valid(added_edges):
        # Return as list of [u, v] lists
        return [[u, v] for u, v in added_edges]
    
    # Fallback: return an empty list (this should not happen)
    return []
# EVOLVE-BLOCK-END
