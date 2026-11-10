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
        
        cross_edges = []
        C_edge_set = set()
        
        for (a, b) in edges:
            if a < k and b < k:
                C_edge_set.add(frozenset((a, b)))
            elif a >= k and b >= k:
                pass
            else:
                if a < k:
                    c, i = a, b
                else:
                    c, i = b, a
                cross_edges.append((i, c))
        
        def cover_C_edges(C, edge_set):
            remaining = set(edge_set)
            cliques = []
            C_list = list(C)
            while remaining:
                best_clique = None
                best_count = 0
                for start in C_list:
                    clique = [start]
                    clique_set = {start}
                    def in_clique_edges(v, cl):
                        return all(frozenset((v, u)) in remaining for u in cl)
                    candidates = [v for v in C_list if v != start and in_clique_edges(v, clique_set)]
                    while candidates:
                        v = candidates.pop(0)
                        clique.append(v)
                        clique_set.add(v)
                        candidates = [w for w in candidates if in_clique_edges(w, clique_set)]
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
                for e in list(remaining):
                    if e.issubset(set(best_clique)):
                        remaining.discard(e)
            return cliques

        def mis_with_used(vertices, used_edges):
            remaining = set(vertices)
            S = set()
            while remaining:
                min_conf = float('inf')
                best_v = None
                for v in remaining:
                    conf = 0
                    for u in remaining:
                        if u != v and frozenset((u,v)) in used_edges:
                            conf += 1
                    if conf < min_conf:
                        min_conf = conf
                        best_v = v
                S.add(best_v)
                to_remove = {best_v}
                for u in remaining:
                    if frozenset((u, best_v)) in used_edges:
                        to_remove.add(u)
                remaining -= to_remove
            return S

        uncovered_cross = set(cross_edges)
        used_CC_edges = set()
        mixed_cliques = []
        
        while uncovered_cross:
            best_i = None
            best_S = None
            best_score = -1
            
            for i in I:
                U_i = set(c for (i_idx, c) in uncovered_cross if i_idx == i)
                if not U_i:
                    continue
                
                S = mis_with_used(list(U_i), used_CC_edges)
                score = len(S)
                
                if score > best_score:
                    best_score = score
                    best_i = i
                    best_S = S
            
            if best_i is None:
                break
            
            clique = [best_i] + list(best_S)
            mixed_cliques.append(clique)
            
            for c in best_S:
                uncovered_cross.discard((best_i, c))
            
            S_list = list(best_S)
            for idx_a in range(len(S_list)):
                for idx_b in range(idx_a + 1, len(S_list)):
                    used_CC_edges.add(frozenset((S_list[idx_a], S_list[idx_b])))

        remaining_CC_edges = C_edge_set - used_CC_edges
        pure_cliques = cover_C_edges(C, remaining_CC_edges)
        
        result = mixed_cliques + pure_cliques
        return result
    
    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(solve(*spec))
    return results
