def construct():
    from random import Random
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)
    
    def solve(n, k, p, seed):
        edges = instance(n, k, p, seed)
        C = list(range(k))
        I = list(range(k, n))
        
        # Build adjacency for cross edges
        cross = {i: set() for i in I}
        for (a, b) in edges:
            if a < k and b < k:
                continue
            elif a >= k and b >= k:
                continue
            else:
                if a < k:
                    c, i = a, b
                else:
                    c, i = b, a
                cross[i].add(c)
        
        # Initialize available edges in C
        # All pairs in C are edges initially
        available_C_edges = set(frozenset((u, v)) for u in range(k) for v in range(u + 1, k))
        
        # Uncovered cross edges
        # Represent as set of (i, c)
        uncovered_cross = set()
        for i in I:
            for c in cross[i]:
                uncovered_cross.add((i, c))
        
        mixed_cliques = []
        
        # Greedy selection of mixed cliques
        # To avoid infinite loops or issues if no progress is made (shouldn't happen)
        while uncovered_cross:
            best_i = None
            best_S = None
            best_count = 0
            
            for i in I:
                # Get neighbors of i that have uncovered cross edges
                # N_i = { c in cross[i] : (i, c) in uncovered_cross }
                N_i = [c for c in cross[i] if (i, c) in uncovered_cross]
                if not N_i:
                    continue
                
                # Find a large clique S in N_i using available_C_edges
                # Greedy Max Clique on the subgraph of available_C_edges induced by N_i
                S = []
                candidates = list(N_i)
                
                # Sort candidates by degree in available graph within candidates?
                # Or just iterate.
                while candidates:
                    # Pick vertex with max degree in available_C_edges restricted to candidates
                    # Degree of v = number of u in candidates such that {v, u} in available_C_edges
                    best_v = None
                    best_deg = -1
                    for v in candidates:
                        deg = 0
                        for u in candidates:
                            if v == u: continue
                            if frozenset((v, u)) in available_C_edges:
                                deg += 1
                        if deg > best_deg:
                            best_deg = deg
                            best_v = v
                    
                    if best_v is None:
                        break
                    
                    S.append(best_v)
                    candidates.remove(best_v)
                    
                    # Remove from candidates any vertex not connected to best_v in available_C_edges
                    new_candidates = []
                    for u in candidates:
                        if frozenset((best_v, u)) in available_C_edges:
                            new_candidates.append(u)
                    candidates = new_candidates
                
                if len(S) > best_count:
                    best_count = len(S)
                    best_i = i
                    best_S = S
            
            if best_i is None:
                # Should not happen if uncovered_cross is not empty
                break
            
            # Create clique
            clique = [best_i] + best_S
            mixed_cliques.append(clique)
            
            # Update uncovered_cross
            for c in best_S:
                uncovered_cross.discard((best_i, c))
            
            # Update available_C_edges
            for idx_a in range(len(best_S)):
                for idx_b in range(idx_a + 1, len(best_S)):
                    e = frozenset((best_S[idx_a], best_S[idx_b]))
                    available_C_edges.discard(e)
        
        # Cover remaining C-edges with pure cliques
        pure_cliques = cover_C_edges(C, available_C_edges)
        
        result = mixed_cliques + pure_cliques
        return result

    def cover_C_edges(C, edge_set):
        remaining = set(edge_set)
        cliques = []
        C_list = list(C)
        
        while remaining:
            best_clique = None
            best_count = 0
            
            # Find maximal cliques in the remaining graph
            # Greedy: pick a vertex, expand
            for start in C_list:
                # Check if start has any remaining edges
                if not any(frozenset((start, v)) in remaining for v in C_list if v != start):
                    continue
                
                clique = [start]
                clique_set = {start}
                candidates = [v for v in C_list if v != start and frozenset((start, v)) in remaining]
                
                while candidates:
                    # Pick candidate with max degree in remaining within candidates
                    best_v = None
                    best_deg = -1
                    for v in candidates:
                        deg = 0
                        for u in candidates:
                            if v == u: continue
                            if frozenset((v, u)) in remaining:
                                deg += 1
                        if deg > best_deg:
                            best_deg = deg
                            best_v = v
                            
                    if best_v is None:
                        break
                    
                    clique.append(best_v)
                    clique_set.add(best_v)
                    candidates.remove(best_v)
                    
                    new_candidates = []
                    for u in candidates:
                        if frozenset((best_v, u)) in remaining:
                            new_candidates.append(u)
                    candidates = new_candidates
                
                # Count remaining edges covered by this clique
                count = 0
                for e in remaining:
                    if e.issubset(clique_set):
                        count += 1
                
                if count > best_count:
                    best_count = count
                    best_clique = list(clique)
            
            if best_count == 0 or best_clique is None:
                # If we can't find a clique, something is wrong
                # But remaining edges must form some structure
                # Fallback: pick any edge and make a clique of size 2
                # This ensures termination
                e = next(iter(remaining))
                u, v = list(e)
                best_clique = [u, v]
                best_count = 1
            
            cliques.append(best_clique)
            
            # Remove covered edges
            clique_set = set(best_clique)
            for e in list(remaining):
                if e.issubset(clique_set):
                    remaining.discard(e)
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(solve(*spec))
    return results
