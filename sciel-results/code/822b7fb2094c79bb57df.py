import time
import itertools

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set(tuple(e) for e in P['edges'])
    base_edges.update((v, u) for u, v in base_edges)
    
    # Generate candidate edges: Only E-O to ensure triangle-free (bipartite)
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in base_edges:
                continue
            # Check parity
            if u % 2 != v % 2:
                candidates.append((u, v))
                
    # We want to find min k such that there exists a subset of k candidates
    # that makes diameter <= 4.
    
    # Pre-compute distances for base graph
    def get_dist_matrix(edges):
        dist = [[float('inf')] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0
        for u, v in edges:
            dist[u][v] = 1
            dist[v][u] = 1
        # Floyd-Warshall
        for k in range(n):
            for i in range(n):
                if dist[i][k] == float('inf'):
                    continue
                for j in range(n):
                    if dist[k][j] == float('inf'):
                        continue
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        return dist

    def max_diameter(dist):
        return max(max(row) for row in dist)

    base_dist = get_dist_matrix(P['edges'])
    
    # Greedy approach
    current_edges = list(P['edges'])
    current_dist = base_dist
    current_diam = max_diameter(current_dist)
    
    added_edges = []
    deadline = time.time() + 50
    
    while current_diam > 4:
        if time.time() > deadline:
            break
        best_edge = None
        best_diam = current_diam
        # Try adding each candidate
        # Optimization: Only try candidates that reduce the max distance
        for u, v in candidates:
            if (u, v) in [(e[0], e[1]) for e in current_edges] or (v, u) in [(e[0], e[1]) for e in current_edges]:
                continue
            # Quick check: does this edge reduce any pair distance?
            # Approximation: if dist[u][v] > 1, it might help.
            # But we need to check global diameter.
            # To speed up, we can just pick the edge that connects the farthest pair's neighbors?
            # Simpler: just iterate.
            
            new_edges = current_edges + [(u, v)]
            new_dist = get_dist_matrix(new_edges)
            new_diam = max_diameter(new_dist)
            
            if new_diam < best_diam:
                best_diam = new_diam
                best_edge = (u, v)
                
        if best_edge is None:
            # No single edge improves diameter?
            # This shouldn't happen if graph is connected and we can add edges.
            # But if we are stuck, we might need to add an edge that doesn't immediately reduce max diam but enables later ones?
            # In diameter reduction, usually any edge that shortens a path helps.
            # If no edge reduces max diam, we are stuck.
            break
            
        current_edges.append(best_edge)
        added_edges.append(list(best_edge))
        current_dist = get_dist_matrix(current_edges)
        current_diam = max_diameter(current_dist)
        
    # Verify triangle-free (bipartite check)
    # Since we only added E-O edges to a bipartite graph, it remains bipartite.
    
    return added_edges
