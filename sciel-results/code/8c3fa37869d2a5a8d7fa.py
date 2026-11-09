"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    
    # Base edges are the path 0-1-2-...-23
    base_edges = set()
    for e in P['edges']:
        base_edges.add(tuple(sorted(e)))
    
    # Build adjacency list for base graph
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Helper: check if adding edge (u,v) keeps graph triangle-free
    def is_triangle_free_add(adj, u, v):
        # If u and v have a common neighbor, adding (u,v) creates a triangle
        common = adj[u] & adj[v]
        return len(common) == 0
    
    # Helper: compute diameter via BFS from each vertex
    def compute_diameter(adj):
        max_dist = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            queue = [src]
            for node in queue:
                for nbr in adj[node]:
                    if dist[nbr] == -1:
                        dist[nbr] = dist[node] + 1
                        queue.append(nbr)
            # If any dist is -1, graph is disconnected
            if -1 in dist:
                return float('inf')
            max_dist = max(max_dist, max(dist))
        return max_dist
    
    # Helper: verify a set of added edges produces valid graph
    def verify_solution(added_edges):
        adj_test = [set(s) for s in adj]
        for u, v in added_edges:
            if not is_triangle_free_add(adj_test, u, v):
                return False
            adj_test[u].add(v)
            adj_test[v].add(u)
        diam = compute_diameter(adj_test)
        return diam <= 4
    
    # Strategy: Use hub-and-spoke with vertex 12 as hub
    # Connect vertex 12 to all even vertices except 12 itself
    # This creates a bipartite graph (triangle-free)
    # Even vertices: 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22
    # Odd vertices: 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23
    
    # Candidate 1: Connect 12 to all even vertices except 12
    candidate1 = []
    for v in [0, 2, 4, 6, 8, 10, 14, 16, 18, 20, 22]:
        if (12, v) not in base_edges and sorted((12, v)) not in base_edges:
            candidate1.append((min(12, v), max(12, v)))
    
    if verify_solution(candidate1):
        return [[int(u), int(v)] for u, v in candidate1]
    
    # Candidate 2: Try different hub vertices
    best_edges = None
    best_count = float('inf')
    
    for hub in range(n):
        # Try connecting hub to all vertices of same parity except itself
        parity = hub % 2
        candidates = []
        for v in range(n):
            if v != hub and v % 2 == parity:
                edge = tuple(sorted((hub, v)))
                if edge not in base_edges:
                    candidates.append(edge)
        
        if verify_solution(candidates) and len(candidates) < best_count:
            best_count = len(candidates)
            best_edges = candidates
    
    if best_edges is not None:
        return [[int(u), int(v)] for u, v in best_edges]
    
    # Fallback: greedy approach
    adj_greedy = [set(s) for s in adj]
    added = []
    
    for iteration in range(20):
        curr_diam = compute_diameter(adj_greedy)
        if curr_diam <= 4:
            break
        
        best_edge = None
        best_new_diam = float('inf')
        
        for u in range(n):
            for v in range(u + 1, n):
                edge = (u, v)
                if edge in base_edges:
                    continue
                if edge in added:
                    continue
                if not is_triangle_free_add(adj_greedy, u, v):
                    continue
                
                adj_greedy[u].add(v)
                adj_greedy[v].add(u)
                new_diam = compute_diameter(adj_greedy)
                adj_greedy[u].remove(v)
                adj_greedy[v].remove(u)
                
                if new_diam < best_new_diam:
                    best_new_diam = new_diam
                    best_edge = edge
        
        if best_edge is None:
            break
        
        u, v = best_edge
        adj_greedy[u].add(v)
        adj_greedy[v].add(u)
        added.append(list([u, v]))
    
    if compute_diameter(adj_greedy) <= 4:
        return [[int(u), int(v)] for u, v in added]
    
    # Last resort: return empty (should not happen)
    return []

# EVOLVE-BLOCK-END
