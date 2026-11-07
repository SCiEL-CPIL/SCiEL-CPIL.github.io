import time
import random

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s kill

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_split(n, k, p, seed):
        edges = instance(n, k, p, seed)
        if not edges:
            return []
        
        # Build adjacency sets
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        cliques = []
        
        C = set(range(k))
        I = set(range(k, n))
        
        # 1. Forced cross cliques: for each v in I with neighbors in C, use {v} ∪ N(v)∩C
        covered_internal_edges = set()
        
        for v in sorted(I):
            neighbors_in_C = adj[v] & C
            if not neighbors_in_C:
                continue
            
            clique_v = [v] + sorted(neighbors_in_C)
            cliques.append(clique_v)
            
            # Mark internal edges within neighbors_in_C as covered
            neigh_list = sorted(neighbors_in_C)
            for i in range(len(neigh_list)):
                for j in range(i + 1, len(neigh_list)):
                    u, w = neigh_list[i], neigh_list[j]
                    e = (u, w) if u < w else (w, u)
                    covered_internal_edges.add(e)
        
        # 2. Handle remaining internal edges within C
        all_internal_edges = set()
        for u in C:
            for w in adj[u] & C:
                if u < w:
                    all_internal_edges.add((u, w))
        
        remaining_edges = all_internal_edges - covered_internal_edges
        
        if remaining_edges:
            # Build subgraph of remaining edges on C
            rem_adj = {u: set() for u in C}
            for u, w in remaining_edges:
                rem_adj[u].add(w)
                rem_adj[w].add(u)
            
            rem_edges_set = set(remaining_edges)
            
            while rem_edges_set:
                if time.time() > deadline:
                    # Fallback: cover remaining edges individually
                    for e in rem_edges_set:
                        cliques.append([e[0], e[1]])
                    break
                
                # Find maximum clique in remaining graph using Bron-Kerbosch
                max_clique = _find_max_clique(rem_adj, rem_edges_set)
                
                if len(max_clique) < 2:
                    u, w = next(iter(rem_edges_set))
                    cliques.append([u, w])
                    rem_edges_set.discard((u, w) if u < w else (w, u))
                    continue
                
                cliques.append(sorted(max_clique))
                
                # Remove edges covered by this clique
                for i in range(len(max_clique)):
                    for j in range(i + 1, len(max_clique)):
                        u, w = max_clique[i], max_clique[j]
                        e = (u, w) if u < w else (w, u)
                        rem_edges_set.discard(e)
        
        return cliques

    def _find_max_clique(adj, edge_set):
        """Find a maximum clique (by size) in the graph defined by adj and edge_set."""
        active = set()
        for u, v in edge_set:
            active.add(u)
            active.add(v)
        if not active:
            return []
        
        best = []
        
        def bron_kerbosch(R, P, X):
            nonlocal best
            if not P and not X:
                if len(R) > len(best):
                    best = list(R)
                return
            
            # Pivot selection
            U = P | X
            if U:
                pivot = max(U, key=lambda v: len(adj.get(v, set()) & P))
                candidates = P - adj.get(pivot, set())
            else:
                candidates = P
            
            for v in list(candidates):
                bron_kerbosch(R | {v}, P & adj.get(v, set()), X & adj.get(v, set()))
                P.discard(v)
                X.add(v)
        
        P = set(active)
        X = set()
        bron_kerbosch(set(), P, X)
        
        return best

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
        else:
            res = solve_split(*spec)
            clean_res = []
            for clique in res:
                clean_clique = [int(v) for v in clique]
                clean_res.append(clean_clique)
            results.append(clean_res)
            
    return results
