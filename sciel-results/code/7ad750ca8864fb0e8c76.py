def construct():
    from random import Random
    import itertools

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)
    
    def solve(n, k, p, seed):
        edges = instance(n, k, p, seed)
        C = list(range(k))
        I = list(range(k, n))
        
        all_C_edges = set(frozenset((u, v)) for u in range(k) for v in range(u+1, k))
        uncovered_C = set(all_C_edges)
        
        uncovered_cross = set()
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
                uncovered_cross.add((i, c))
                
        mixed_cliques = []
        
        while uncovered_cross:
            degs = {}
            for (i, c) in uncovered_cross:
                degs[i] = degs.get(i, 0) + 1
            
            if not degs:
                break
                
            i_max = max(degs, key=lambda x: degs[x])
            S = [c for (i, c) in uncovered_cross if i == i_max]
            
            clique = [i_max] + S
            mixed_cliques.append(clique)
            
            to_remove_cross = set((i_max, c) for c in S)
            uncovered_cross -= to_remove_cross
            
            S_set = set(S)
            to_remove_C = []
            for e in uncovered_C:
                if e.issubset(S_set):
                    to_remove_C.append(e)
            for e in to_remove_C:
                uncovered_C.discard(e)
                
        pure_cliques = cover_C_edges(C, uncovered_C)
        
        result = mixed_cliques + pure_cliques
        return result

    def cover_C_edges(C, edge_set):
        remaining = set(edge_set)
        cliques = []
        C_list = list(C)
        
        while remaining:
            best_clique = None
            best_count = 0
            
            # Try starting from each vertex
            for start in C_list:
                # Build a maximal clique greedily
                clique = [start]
                clique_set = {start}
                
                # Candidates: vertices connected to all in clique
                # Start with all
                candidates = [v for v in C_list if v != start]
                
                # Filter candidates that are connected to start
                candidates = [v for v in candidates if frozenset((v, start)) in remaining]
                
                while candidates:
                    # Pick candidate with max degree in remaining
                    best_v = None
                    best_deg = -1
                    for v in candidates:
                        # Check connectivity to all in clique
                        ok = True
                        for u in clique:
                            if u == v: continue
                            if frozenset((v, u)) not in remaining:
                                ok = False
                                break
                        if not ok:
                            continue
                        
                        deg = 0
                        for e in remaining:
                            if v in e:
                                deg += 1
                        if deg > best_deg:
                            best_deg = deg
                            best_v = v
                            
                    if best_v is None:
                        break
                        
                    clique.append(best_v)
                    clique_set.add(best_v)
                    
                    # Update candidates: must be connected to best_v
                    new_candidates = []
                    for v in candidates:
                        if v == best_v:
                            continue
                        if frozenset((v, best_v)) in remaining:
                            new_candidates.append(v)
                    candidates = new_candidates
                            
                # Count edges covered
                count = 0
                for e in remaining:
                    if e.issubset(clique_set):
                        count += 1
                            
                if count > best_count:
                    best_count = count
                    best_clique = list(clique)
                    
            if best_count == 0 or best_clique is None:
                break
                
            cliques.append(best_clique)
            clique_set = set(best_clique)
            to_remove = []
            for e in remaining:
                if e.issubset(clique_set):
                    to_remove.append(e)
            for e in to_remove:
                remaining.discard(e)
                
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(solve(*spec))
    return results
