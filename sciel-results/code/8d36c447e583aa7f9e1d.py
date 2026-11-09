import random
import itertools
from functools import lru_cache

def construct():
    n = 18
    
    # Define the C4-free bipartite graph structure
    # Based on Fano plane lines truncated to 6 points
    # Points 0..5
    # Lines (subsets of points):
    lines = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5],
        [1, 3, 5],
        [1, 4],
        [2, 3]
    ]
    
    def get_bipartite_edges(base_a, base_b):
        edges = []
        for i, line in enumerate(lines):
            for p in line:
                edges.append([base_a + p, base_b + i])
        return edges

    # Parts: A=0..5, B=6..11, C=12..17
    edges = []
    
    # Internal matchings
    for base in [0, 6, 12]:
        for i in range(3):
            edges.append([base + 2*i, base + 2*i + 1])
            
    # Cross edges
    # A-B
    edges.extend(get_bipartite_edges(0, 6))
    # B-C
    edges.extend(get_bipartite_edges(6, 12))
    # C-A
    edges.extend(get_bipartite_edges(12, 0))
    
    # Build adjacency
    adj = [[False]*n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        
    # Check K4 and remove edges if necessary
    # We will iteratively find K4s and remove an edge to break them
    # Prioritize removing cross edges to maintain sparsity? Or just any edge.
    
    def find_k4():
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i][j]: continue
                for k in range(j+1, n):
                    if not adj[i][k] or not adj[j][k]: continue
                    for l in range(k+1, n):
                        if adj[i][l] and adj[j][l] and adj[k][l]:
                            return (i, j, k, l)
        return None
        
    # Iteratively remove edges to ensure K4-free
    # Limit iterations to avoid infinite loops or timeouts
    for _ in range(100):
        k4 = find_k4()
        if not k4:
            break
        i, j, k, l = k4
        # Remove edge (i, j)
        adj[i][j] = False
        adj[j][i] = False
        # Remove from edges list
        # To keep edges list consistent, we'll rebuild it from adj at the end
        # Or just mark it. Since we rebuild, we can just modify adj.
        
    # Rebuild edges from adj
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append([i, j])
                
    # Compute max triangle-free induced subgraph
    def is_triangle_free(vertex_set):
        vs = list(vertex_set)
        for i in range(len(vs)):
            for j in range(i+1, len(vs)):
                for k in range(j+1, len(vs)):
                    u, v, w = vs[i], vs[j], vs[k]
                    if adj[u][v] and adj[v][w] and adj[u][w]:
                        return False
        return True
        
    def max_triangle_free():
        best = 0
        # Check sizes from 13 down to 10
        for size in range(13, 9, -1):
            found = False
            for comb in itertools.combinations(range(n), size):
                if is_triangle_free(comb):
                    return size
        # If not found above 9, check 9
        for comb in itertools.combinations(range(n), 9):
            if is_triangle_free(comb):
                return 9
        return 8

    score = max_triangle_free()
    
    return edges
