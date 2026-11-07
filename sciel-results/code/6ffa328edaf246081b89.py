"""Combined program with optimized split-graph clique partition."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_split(n, k, p, seed):
        start_time = time.time()
        deadline = start_time + 110
        
        rng = random.Random(seed)
        
        # Generate cross edges and neighborhoods
        N = [set() for _ in range(n)]
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))
                    N[v].add(u)
        
        # Baseline: 1 clique for C + 1 clique per cross edge
        # Try to improve by using larger cliques for I-vertices
        
        # Strategy: Greedily select which I-vertices to make "active" (use {v} ∪ N[v])
        # and which to keep "inactive" (use individual 2-cliques)
        # Active vertices must have pairwise intersection ≤ 1
        
        # Sort I-vertices by degree (larger first) for greedy selection
        i_vertices = list(range(k, n))
        i_vertices.sort(key=lambda v: len(N[v]), reverse=True)
        
        # Greedy: select active vertices
        active = set()
        covered_c_pairs = set()  # C-edges covered by active vertices
        
        for v in i_vertices:
            if time.time() > deadline:
                break
            nv = list(N[v])
            # Check if adding v as active conflicts
            conflict = False
            for i in range(len(nv)):
                for j in range(i + 1, len(nv)):
                    pair = (min(nv[i], nv[j]), max(nv[i], nv[j]))
                    if pair in covered_c_pairs:
                        conflict = True
                        break
                if conflict:
                    break
            if not conflict:
                active.add(v)
                for i in range(len(nv)):
                    for j in range(i + 1, len(nv)):
                        covered_c_pairs.add((min(nv[i], nv[j]), max(nv[i], nv[j])))
        
        # Build solution
        cliques = []
        
        # Add cliques for active vertices
        for v in active:
            cliques.append([v] + sorted(N[v]))
        
        # Add 2-cliques for inactive vertices
        for v in range(k, n):
            if v not in active:
                for u in N[v]:
                    cliques.append([u, v])
        
        # Cover remaining C-edges
        all_c_pairs = set()
        for u in range(k):
            for w in range(u + 1, k):
                all_c_pairs.add((u, w))
        
        remaining = all_c_pairs - covered_c_pairs
        
        if remaining:
            # Greedy clique cover on the remaining C-edges
            rem_adj = {u: set() for u in range(k)}
            for u, w in remaining:
                rem_adj[u].add(w)
                rem_adj[w].add(u)
            
            rem_copy = set(remaining)
            while rem_copy:
                if time.time() > deadline:
                    for u, w in rem_copy:
                        cliques.append([u, w])
                    break
                
                # Find a maximal clique in the remaining graph
                # Use greedy: pick vertex with max degree, expand
                degrees = {u: len([w for w in rem_adj[u] if (min(u,w), max(u,w)) in rem_copy]) for u in range(k)}
                if max(degrees.values(), default=0) == 0:
                    # No edges, should not happen
                    break
                start = max(degrees, key=degrees.get)
                clique = [start]
                candidates = [c for c in rem_adj[start] if (min(start,c), max(start,c)) in rem_copy]
                for c in candidates:
                    if all((min(c, u), max(c, u)) in rem_copy for u in clique):
                        clique.append(c)
                
                cliques.append(sorted(clique))
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        rem_copy.discard((min(clique[i], clique[j]), max(clique[i], clique[j])))
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        cliques = solve_split(n, k, p, seed)
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END
