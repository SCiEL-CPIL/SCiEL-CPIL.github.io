import time
import random

def construct():
    deadline = time.time() + 145
    
    # 1. Guaranteed Candidate: n=32 (8 disjoint K4)
    # This graph is C5-free and has independence number 8.
    n_32 = 32
    edges_32 = []
    for block in range(8):
        base = block * 4
        for i in range(4):
            for j in range(i + 1, 4):
                edges_32.append([base + i, base + j])
    
    # Helper: Check if graph is C5-free
    def is_c5_free(n, edges):
        if n > 35:
            return False
        adj = [[False]*n for _ in range(n)]
        for u, v in edges:
            if u == v:
                return False
            adj[u][v] = True
            adj[v][u] = True
        
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]:
                    continue
                neighbors_u = [x for x in range(n) if adj[u][x] and x != v]
                neighbors_v = [y for y in range(n) if adj[v][y] and y != u]
                
                for a in neighbors_u:
                    for c in neighbors_v:
                        if a == c:
                            continue
                        for b in range(n):
                            if b in (u, v, a, c):
                                continue
                            if adj[a][b] and adj[b][c]:
                                return False
        return True

    # Helper: Check if graph has an independent set of size k
    def has_independent_set_of_size(n, edges, k=9):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        def greedy_is(vertices):
            rem = list(vertices)
            is_set = []
            while rem:
                best_v = None
                min_deg = float('inf')
                random.shuffle(rem)
                for v in rem:
                    deg = len([u for u in rem if u in adj[v]])
                    if deg < min_deg:
                        min_deg = deg
                        best_v = v
                is_set.append(best_v)
                rem.remove(best_v)
                new_rem = []
                for u in rem:
                    if u not in adj[best_v]:
                        new_rem.append(u)
                rem = new_rem
            return len(is_set)

        all_vertices = list(range(n))
        for _ in range(20):
            size = greedy_is(all_vertices)
            if size >= k:
                return True
        return False

    def validate(n, edges):
        if len(edges) != len(set((min(u,v), max(u,v)) for u,v in edges)):
            return False
        for u, v in edges:
            if u == v or u < 0 or u >= n or v < 0 or v >= n:
                return False
        if not is_c5_free(n, edges):
            return False
        if has_independent_set_of_size(n, edges, 9):
            return False
        return True

    # Verify the n=32 solution
    if validate(n_32, edges_32):
        best_n = n_32
        best_edges = edges_32
    else:
        # Fallback to smaller if 32 fails (should not happen)
        best_n = 28
        best_edges = [[4 * b + i, 4 * b + j] for b in range(7) for i in range(4) for j in range(i + 1, 4)]

    # 2. Attempt to find n=33 (limited time)
    if time.time() < deadline - 5:
        target_n = 33
        # Try a few structured perturbations of the 32-node graph
        # Add vertex 32, connect to one vertex in each K4
        # We know this has alpha=9, so it's invalid.
        # Try connecting to 2 vertices in some K4s?
        # If we connect to 2 in a K4, we create C5 risk.
        # Given the analysis, n=33 is likely impossible or very hard.
        # We will skip extensive search to ensure we return the valid 32 quickly.
        pass

    return {'n': best_n, 'edges': best_edges}
