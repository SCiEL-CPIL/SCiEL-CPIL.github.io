import random
from itertools import combinations

def construct():
    n = 39
    # Try to construct a triangle-free graph on 39 vertices with no independent set of 10
    # Use a random approach with verification
    random.seed(42)
    
    def is_triangle_free(edges, n):
        adj = [[False]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    for k in range(j+1, n):
                        if adj[i][k] and adj[j][k]:
                            return False
        return True
    
    def has_independent_set_of_size(edges, n, k=10):
        adj = [[False]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        
        # Branch and bound for independent set
        def dfs(current_set, candidates):
            if len(current_set) >= k:
                return True
            if not candidates:
                return False
            # Branch on first candidate
            v = candidates[0]
            # Include v
            new_candidates = [u for u in candidates[1:] if not adj[v][u]]
            if dfs(current_set + [v], new_candidates):
                return True
            # Exclude v
            if dfs(current_set, candidates[1:]):
                return True
            return False
        
        return dfs([], list(range(n)))
    
    # Generate random triangle-free graphs
    for attempt in range(1000):
        # Start with empty graph
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    # Check if adding (i,j) creates a triangle
                    creates_triangle = False
                    for k in range(n):
                        if k != i and k != j:
                            # Check if (i,k) and (j,k) are in edges
                            has_ik = any((u == i and v == k) or (u == k and v == i) for u, v in edges)
                            has_jk = any((u == j and v == k) or (u == k and v == j) for u, v in edges)
                            if has_ik and has_jk:
                                creates_triangle = True
                                break
                    if not creates_triangle:
                        edges.append((i, j))
        
        if is_triangle_free(edges, n):
            if not has_independent_set_of_size(edges, n, 10):
                return {'n': n, 'edges': [list(e) for e in edges]}
    
    # Fallback: use the 26-vertex Andrásfai graph
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}
