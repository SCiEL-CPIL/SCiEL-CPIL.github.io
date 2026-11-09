import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build adjacency sets
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        C = list(range(k))
        I = list(range(k, n))
        C_set = set(C)
        
        # 1. Form cliques for each I vertex
        cliques = []
        N_v_list = {} 
        for v in I:
            N_v = adj[v] & C_set
            N_v_list[v] = N_v
            if N_v:
                cliques.append(sorted(list(N_v) + [v]))
            
        # 2. Determine covered C-edges
        # Precompute N_u for u in C
        N_u_list = {u: set() for u in C}
        for v in I:
            for u in N_v_list[v]:
                N_u_list[u].add(v)
                
        covered_edges = set()
        for u in C:
            for w in C:
                if u < w:
                    if N_u_list[u] & N_u_list[w]:
                        covered_edges.add((u, w))
                        
        # 3. Determine remaining C-edges
        rem_edges = set()
        for u in C:
            for w in C:
                if u < w:
                    if (u, w) not in covered_edges:
                        rem_edges.add((u, w))
                        
        # 4. Cover remaining C-edges with cliques
        rem_adj = {u: set() for u in C}
        for u, w in rem_edges:
            rem_adj[u].add(w)
            rem_adj[w].add(u)
            
        remaining = set(rem_edges)
        while remaining:
            # Calculate degrees in remaining graph
            dyn_deg = {u: 0 for u in C}
            for u, w in remaining:
                dyn_deg[u] += 1
                dyn_deg[w] += 1
                
            max_deg = -1
            start_v = None
            for u in C:
                if dyn_deg[u] > max_deg:
                    max_deg = dyn_deg[u]
                    start_v = u
                    
            if start_v is None:
                break
                
            # Greedily build clique from start_v
            clique = [start_v]
            # Candidates are neighbors in rem_adj such that edge is in remaining
            candidates = [w for w in rem_adj[start_v] if (min(start_v, w), max(start_v, w)) in remaining]
            candidates.sort(key=lambda x: -dyn_deg[x])
            
            for w in candidates:
                ok = True
                for u in clique:
                    e = (min(u, w), max(u, w))
                    if e not in remaining:
                        ok = False
                        break
                if ok:
                    clique.append(w)
                    
            if len(clique) >= 2:
                cliques.append(sorted(clique))
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        e = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                        remaining.discard(e)
            else:
                break

        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > deadline:
            # Fallback: trivial partition (each edge is a clique)
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue
            
        res = solve_instance(*spec)
        results.append(res)

    return results
