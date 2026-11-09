"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    N = 512
    
    # Build adjacency: a~b iff lcm(a,b) <= N
    n = N
    adj = [set() for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            g = gcd(i, j)
            if (i // g) * j <= N:
                adj[i].add(j)
                adj[j].add(i)
    
    # Maximum clique with branch and bound + greedy coloring
    vertices = list(range(1, n + 1))
    vertices.sort(key=lambda v: len(adj[v]), reverse=True)
    m = len(vertices)
    
    # Adjacency in sorted indices
    adj_idx = [set() for _ in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            vi, vj = vertices[i], vertices[j]
            if vj in adj[vi]:
                adj_idx[i].add(j)
                adj_idx[j].add(i)
    
    best_clique = []
    best_size = 0
    
    def color_bound(cands):
        if not cands:
            return 0
        # Order by degree in cands
        deg = {v: sum(1 for u in adj_idx[v] if u in cands) for v in cands}
        ordered = sorted(cands, key=lambda v: deg[v])
        colors = {}
        max_c = 0
        for v in ordered:
            used = set()
            for u in adj_idx[v]:
                if u in colors:
                    used.add(colors[u])
            c = 0
            while c in used:
                c += 1
            colors[v] = c
            max_c = max(max_c, c)
        return max_c + 1
    
    def expand(clique, cands):
        nonlocal best_clique, best_size
        if time.time() > deadline:
            return
        if not cands:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = list(clique)
            return
        ub = color_bound(cands)
        if len(clique) + ub <= best_size:
            return
        
        # Order by degree
        deg = {v: sum(1 for u in adj_idx[v] if u in cands) for v in cands}
        ordered = sorted(cands, key=lambda v: deg[v], reverse=True)
        
        for v in ordered:
            if time.time() > deadline:
                return
            new_clique = clique + [v]
            new_cands = set(cands)
            new_cands.discard(v)
            # Only keep neighbors of v
            new_cands = {u for u in new_cands if u in adj_idx[v]}
            expand(new_clique, new_cands)
            cands.discard(v)
            if not cands:
                break
    
    expand([], set(range(m)))
    
    result = [vertices[v] for v in best_clique]
    return result if result else [1]
# EVOLVE-BLOCK-END
