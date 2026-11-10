"""Rewritten program for clique partition of split graphs."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, p, seed):
        """
        Structure-aware clique partition for split graphs.
        
        Key insight: In a split graph, any clique contains at most one vertex from I
        (since I is independent). So cliques are either:
        - {i} ∪ S where i ∈ I and S ⊆ C forms a clique in C
        - S ⊆ C forms a clique in C
        
        Strategy:
        1. For each i ∈ I, find a large subset S of N(i) that forms a clique in the
           "uncovered" C-edge graph
        2. Process I-vertices in order of decreasing |N(i)| to maximize coverage
        3. Cover remaining C-edges with C-only cliques
        """
        edges = instance(n, k, p, seed)
        
        # Build N[i] = set of C-vertices adjacent to I-vertex i
        N = {}
        for i in range(k, n):
            N[i] = set()
        for u, v in edges:
            if u < k and v >= k:
                N[v].add(u)
            elif v < k and u >= k:
                N[u].add(v)
        
        # C-edges: all pairs within C
        C_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                C_edges.add((u, v))
        
        # Track which C-edges are covered
        covered_C_edges = set()
        cliques = []
        
        # Process I-vertices in order of decreasing |N(i)|
        I_vertices = sorted(range(k, n), key=lambda i: len(N[i]), reverse=True)
        
        for i in I_vertices:
            if time.time() > deadline:
                break
            if not N[i]:
                continue
            
            remaining_N = set(N[i])
            
            while remaining_N:
                if time.time() > deadline:
                    break
                
                # Find maximum clique in remaining_N where all pairs are uncovered C-edges
                # Use greedy approach: sort by degree in the uncovered subgraph
                # Build adjacency within remaining_N based on uncovered C-edges
                adj = {v: set() for v in remaining_N}
                for v in remaining_N:
                    for w in remaining_N:
                        if v < w:
                            e = (v, w)
                            if e not in covered_C_edges:
                                adj[v].add(w)
                                adj[w].add(v)
                
                # Greedy maximum clique: pick vertex with max degree, add to clique,
                # remove non-neighbors, repeat
                S = set()
                candidates = set(remaining_N)
                while candidates:
                    if time.time() > deadline:
                        break
                    # Pick vertex with max degree in candidates
                    v = max(candidates, key=lambda x: len(adj[x] & candidates))
                    S.add(v)
                    # Keep only neighbors of v in candidates
                    candidates = candidates & adj[v]
                
                if not S:
                    S = {next(iter(remaining_N))}
                
                clique = sorted([i] + list(S))
                cliques.append(clique)
                
                # Mark C-edges within S as covered
                S_list = sorted(S)
                for a_idx in range(len(S_list)):
                    for b_idx in range(a_idx + 1, len(S_list)):
                        e = (S_list[a_idx], S_list[b_idx])
                        covered_C_edges.add(e)
                
                remaining_N -= S
        
        # Cover remaining C-edges with C-only cliques
        remaining_C_edges = C_edges - covered_C_edges
        
        if remaining_C_edges:
            # Build adjacency for remaining C-edges
            remaining_adj = defaultdict(set)
            for u, v in remaining_C_edges:
                remaining_adj[u].add(v)
                remaining_adj[v].add(u)
            
            while remaining_C_edges:
                if time.time() > deadline:
                    break
                
                if not remaining_adj:
                    break
                
                # Find maximum clique in remaining graph
                # Use greedy: pick vertex with max degree, expand to clique
                v = max(remaining_adj.keys(), key=lambda x: len(remaining_adj[x]))
                
                # Build clique starting from v
                S = {v}
                candidates = set(remaining_adj[v])
                while candidates:
                    if time.time() > deadline:
                        break
                    # Pick vertex with max degree in candidates ∩ remaining_adj
                    w = max(candidates, key=lambda x: len(remaining_adj[x] & candidates))
                    S.add(w)
                    # Keep only common neighbors
                    candidates = candidates & remaining_adj[w]
                
                S_list = sorted(S)
                if len(S_list) < 2:
                    if remaining_adj[v]:
                        w = next(iter(remaining_adj[v]))
                        S_list = sorted([v, w])
                    else:
                        break
                
                cliques.append(S_list)
                
                # Remove all edges within S
                for a_idx in range(len(S_list)):
                    for b_idx in range(a_idx + 1, len(S_list)):
                        remaining_C_edges.discard((S_list[a_idx], S_list[b_idx]))
                
                # Update adjacency
                for u in S_list:
                    remaining_adj.pop(u, None)
                    for w in remaining_adj:
                        remaining_adj[w].discard(u)
                remaining_adj = {u: v for u, v in remaining_adj.items() if v}
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            # Fallback: one edge per clique
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            results.append([list(e) for e in edges])
            continue
        cliques = partition_split_graph(*spec)
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END
