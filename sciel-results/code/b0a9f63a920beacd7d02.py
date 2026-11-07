import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    def instance(n, k, p, seed):
        import random
        rng = random.Random(seed)
        edges = []
        # Clique on first k
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        # Independent set on rest, cross edges with prob p
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_one(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build adjacency for C and I
        # C is 0..k-1, I is k..n-1
        adj_C = [set() for _ in range(k)]
        adj_I_to_C = [set() for _ in range(n)] # Only store neighbors in C for I vertices
        
        # Edges within C are all present
        for u in range(k):
            for v in range(u + 1, k):
                adj_C[u].add(v)
                adj_C[v].add(u)
                
        # Cross edges
        for u, v in edges:
            if u < k and v >= k:
                adj_I_to_C[v].add(u)
            elif v < k and u >= k:
                adj_I_to_C[u].add(v)
        
        # G_rem: set of edges in C not yet covered
        # Represent as set of tuples (u, v) with u < v
        G_rem = set()
        for u in range(k):
            for v in range(u + 1, k):
                G_rem.add((u, v))
                
        cliques = []
        
        # Process each v in I
        # Sort by degree in C descending to cover more edges earlier?
        i_vertices = list(range(k, n))
        i_vertices.sort(key=lambda x: len(adj_I_to_C[x]), reverse=True)
        
        for v in i_vertices:
            if time.time() > deadline:
                break
            N_v = adj_I_to_C[v]
            if not N_v:
                continue
                
            # Find max clique S in G_rem induced by N_v
            # Vertices are in N_v. Edges are in G_rem.
            # Brute force for small k?
            # k is up to 30. 2^30 is too big.
            # But G_rem shrinks.
            # Use a simple greedy or backtracking for max clique in the induced subgraph.
            
            vertices = list(N_v)
            # Filter vertices that actually have edges in G_rem?
            # If a vertex has no edges in G_rem to other vertices in N_v, it can't be in a clique of size > 1.
            
            # Simple greedy:
            # Try to build a clique.
            # Since we want max size, and k is small, we can try all subsets if |N_v| is small?
            # No, |N_v| can be up to 30.
            
            # Heuristic: 
            # 1. If all pairs in N_v are in G_rem, S = N_v.
            # 2. Else, find a large clique.
            
            # Check if N_v induces a clique in G_rem
            is_clique = True
            for u1 in N_v:
                for u2 in N_v:
                    if u1 == u2: continue
                    if u1 < u2:
                        if (u1, u2) not in G_rem:
                            is_clique = False
                            break
                if not is_clique:
                    break
            
            if is_clique and len(N_v) >= 2:
                S = N_v
            else:
                # Find max clique in G_rem induced by N_v
                # Use a simple branch and bound or greedy
                # Given time constraints, a greedy approach:
                # Start with empty set, add vertices one by one if they connect to all in set.
                # Order by degree in G_rem within N_v.
                
                degs = {}
                for u in N_v:
                    count = 0
                    for w in N_v:
                        if u != w and (min(u, w), max(u, w)) in G_rem:
                            count += 1
                    degs[u] = count
                    
                sorted_v = sorted(N_v, key=lambda x: degs[x], reverse=True)
                S = set()
                for u in sorted_v:
                    # Check if u connects to all in S
                    can_add = True
                    for w in S:
                        if (min(u, w), max(u, w)) not in G_rem:
                            can_add = False
                            break
                    if can_add:
                        S.add(u)
                        
            if len(S) >= 2:
                cliques.append([v] + list(S))
                # Remove covered edges from G_rem
                for u, w in combinations(S, 2):
                    G_rem.discard((min(u, w), max(u, w)))
            
            # Cover remaining edges (v, u) for u in N_v \ S
            for u in N_v:
                if u not in S:
                    cliques.append([v, u])
        
        # Cover remaining edges in G_rem
        # They are edges in C.
        # Just add them as cliques of size 2.
        # Or try to merge?
        # Since they are likely few, just add as edges.
        for u, w in G_rem:
            cliques.append([u, w])
            
        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    result = []
    for spec in specs:
        if time.time() > deadline:
            # Fallback to seed if running out of time
            def seed_solve(spec):
                n, k, p, seed = spec
                edges = instance(n, k, p, seed)
                return [list(e) for e in edges]
            result.append(seed_solve(spec))
        else:
            result.append(solve_one(*spec))
            
    return result
