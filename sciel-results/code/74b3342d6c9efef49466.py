import time
from itertools import combinations

def construct():
    from random import Random
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def min_clique_partition(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        S = {}
        for v in range(k, n):
            S[v] = adj[v] & set(range(k))
            
        C = {v: [set(S[v])] for v in range(k, n)}
        
        # Edge to owner map: edge -> v
        edge_owner = {}
        
        def add_edges(v, edge_set):
            for e in edge_set:
                edge_owner[e] = v
                
        def remove_edges(v, edge_set):
            for e in edge_set:
                if edge_owner.get(e) == v:
                    del edge_owner[e]
                    
        # Initialize
        for v in range(k, n):
            subset = C[v][0]
            if len(subset) >= 2:
                es = set()
                for u1, u2 in combinations(subset, 2):
                    es.add((min(u1, u2), max(u1, u2)))
                add_edges(v, es)
                
        # Resolve conflicts
        # We need to detect conflicts.
        # If we maintain edge_owner, a conflict means an edge is claimed by two v's?
        # No, edge_owner only stores ONE owner.
        # So if we try to add an edge already in edge_owner, that's a conflict.
        
        # But we initialized with potential conflicts.
        # So we need to find them first.
        
        # Let's re-initialize by checking for conflicts during addition?
        # No, let's just scan.
        
        # Better: Maintain a list of conflicts?
        # Or just iterate:
        # 1. Find an edge that is "double covered".
        # But edge_owner only has one.
        # So we need to know which edges are covered by which v.
        
        # Let's use a dict: edge -> set of v's
        edge_owners = {}
        
        def compute_owners():
            edge_owners.clear()
            for v in range(k, n):
                for subset in C[v]:
                    if len(subset) >= 2:
                        for u1, u2 in combinations(subset, 2):
                            e = (min(u1, u2), max(u1, u2))
                            if e not in edge_owners:
                                edge_owners[e] = set()
                            edge_owners[e].add(v)
                            
        compute_owners()
        
        while True:
            # Find conflict
            conflict_edge = None
            for e, owners in edge_owners.items():
                if len(owners) > 1:
                    conflict_edge = e
                    break
                    
            if not conflict_edge:
                break
                
            owners = edge_owners[conflict_edge]
            # Pick one to split
            v_to_split = list(owners)[0]
            
            u, w = conflict_edge
            # Find subset in C[v_to_split] containing u and w
            found = False
            new_C_v = []
            for subset in C[v_to_split]:
                if u in subset and w in subset:
                    # Split
                    A = subset - {w}
                    B = {w}
                    new_C_v.append(A)
                    new_C_v.append(B)
                    found = True
                    break
                else:
                    new_C_v.append(subset)
            
            if found:
                C[v_to_split] = new_C_v
                # Recompute owners is expensive.
                # But n is small.
                compute_owners()
            else:
                break
                
        # Calculate remaining edges
        all_K_edges = set((u, v) for u in range(k) for v in range(u+1, k))
        covered_K_edges = set(edge_owners.keys())
        rem_K_edges = all_K_edges - covered_K_edges
        
        rem_cliques = []
        remaining = list(rem_K_edges)
        
        while remaining:
            verts = set()
            for u, v in remaining:
                verts.add(u)
                verts.add(v)
                
            # Greedy Maximal Clique
            cand = set(verts)
            cliq = set()
            while cand:
                # Pick vertex with max degree in remaining graph induced by cand
                degs = {v: 0 for v in cand}
                for u, v in remaining:
                    if u in cand and v in cand:
                        degs[u] += 1
                        degs[v] += 1
                v = max(cand, key=lambda x: degs[x])
                cliq.add(v)
                new_cand = set()
                for u in cand:
                    if u != v and (min(u, v), max(u, v)) in remaining:
                        new_cand.add(u)
                cand = new_cand
                
            rem_cliques.append(sorted(cliq))
            
            covered_in_clique = set()
            for u, v in combinations(cliq, 2):
                covered_in_clique.add((min(u, v), max(u, v)))
                
            remaining = [e for e in remaining if e not in covered_in_clique]
            
        result = []
        for v in range(k, n):
            for subset in C[v]:
                if len(subset) >= 2:
                    result.append(sorted([v] + list(subset)))
                elif len(subset) == 1:
                    u = list(subset)[0]
                    result.append(sorted([v, u]))
                
        for clique in rem_cliques:
            if len(clique) >= 2:
                result.append(clique)
                
        return result

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    for spec in specs:
        results.append(min_clique_partition(*spec))
        
    return results
