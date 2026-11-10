"""Clique partition of split graphs using structural properties."""
import time
from random import Random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, edges):
        # Build adjacency
        adj = [set() for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((min(u, v), max(u, v)))
        
        # C: 0..k-1, I: k..n-1
        # Key: C is a clique, I is independent
        # Each clique has at most one I-vertex
        
        # Strategy: 
        # 1. For each I-vertex, create clique {i} + (its C-neighbors)
        #    But to avoid covering C-C edges twice, we need to be careful.
        # 2. Cover remaining C-C edges with one clique.
        
        # Better strategy: 
        # Process I-vertices. For each, select a subset of C-neighbors
        # such that no C-C edge within the subset is already "claimed" by
        # a previous I-vertex's clique.
        
        i_vertices = list(range(k, n))
        i_vertices.sort(key=lambda i: len([c for c in adj[i] if c < k]), reverse=True)
        
        covered = set()
        cliques = []
        
        for i in i_vertices:
            c_neigh = [c for c in adj[i] if c < k]
            if not c_neigh:
                continue
            
            # Greedily select C-vertices avoiding covered C-C edges
            S = []
            for c in c_neigh:
                ok = True
                for c2 in S:
                    e = (min(c, c2), max(c, c2))
                    if e in covered:
                        ok = False
                        break
                if ok:
                    S.append(c)
            
            if S:
                clique = S + [i]
                cliques.append(clique)
                for idx, u in enumerate(clique):
                    for v in clique[idx+1:]:
                        covered.add((min(u, v), max(u, v)))
        
        # Remaining C-C edges
        rem = edge_set - covered
        if rem:
            verts = set()
            for u, v in rem:
                verts.add(u)
                verts.add(v)
            if len(verts) >= 2:
                cliques.append(list(verts))
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            break
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        cliques = partition_split_graph(n, k, edges)
        cleaned_cliques = []
        for c in cliques:
            cleaned_cliques.append([int(x) for x in c])
        results.append(cleaned_cliques)
    
    return results
# EVOLVE-BLOCK-END
