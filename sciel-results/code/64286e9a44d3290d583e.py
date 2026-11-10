def construct():
    from random import Random
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)
    
    def solve(n, k, p, seed):
        edges = instance(n, k, p, seed)
        # Build adjacency
        # C = 0..k-1, I = k..n-1
        C = list(range(k))
        I = list(range(k, n))
        # cross-edges: (i, c) with i in I, c in C
        cross = {i: [] for i in I}
        C_edge_set = set()
        for (a, b) in edges:
            if a < k and b < k:
                C_edge_set.add(frozenset((a, b)))
            elif a >= k and b >= k:
                # I-internal, shouldn't happen
                pass
            else:
                # cross
                if a < k:
                    c, i = a, b
                else:
                    c, i = b, a
                cross[i].append(c)
        # all C-edges
        all_C_edges = set(frozenset((u, v)) for u in range(k) for v in range(u+1, k))
        used_C_edges = set()
        mixed_cliques = []
        # process I vertices sorted by degree descending
        for i in sorted(I, key=lambda x: -len(cross[x])):
            neighbors = cross[i]
            matched = set()
            # greedy matching on neighbors using unused C-edges
            # to maximize pairs, try to match
            for idx_a in range(len(neighbors)):
                if neighbors[idx_a] in matched:
                    continue
                c_a = neighbors[idx_a]
                # find a partner
                for idx_b in range(idx_a + 1, len(neighbors)):
                    if neighbors[idx_b] in matched:
                        continue
                    c_b = neighbors[idx_b]
                    e = frozenset((c_a, c_b))
                    if e not in used_C_edges:
                        matched.add(c_a)
                        matched.add(c_b)
                        used_C_edges.add(e)
                        mixed_cliques.append([i, c_a, c_b])
                        break
            # singles
            for c in neighbors:
                if c not in matched:
                    mixed_cliques.append([i, c])
        # remaining C-edges
        remaining_C_edges = all_C_edges - used_C_edges
        # cover with pure-C cliques
        pure_cliques = cover_C_edges(C, remaining_C_edges)
        result = mixed_cliques + pure_cliques
        # ensure each clique is a list of ints, >=2 distinct
        return result
    
    def cover_C_edges(C, edge_set):
        remaining = set(edge_set)
        cliques = []
        C_list = list(C)
        while remaining:
            best_clique = None
            best_count = 0
            for start in C_list:
                # greedy maximal clique in graph of remaining edges
                clique = [start]
                clique_set = {start}
                # candidates adjacent to all in clique via remaining
                def in_clique_edges(v, cl):
                    return all(frozenset((v, u)) in remaining for u in cl)
                candidates = [v for v in C_list if v != start and in_clique_edges(v, clique_set)]
                while candidates:
                    # pick candidate with max degree in remaining within candidates+clique (greedy)
                    # just pick first for simplicity, or best
                    v = candidates.pop(0)
                    clique.append(v)
                    clique_set.add(v)
                    candidates = [w for w in candidates if in_clique_edges(w, clique_set)]
                # count remaining edges within clique
                count = 0
                for e in remaining:
                    if e.issubset(clique_set):
                        count += 1
                if count > best_count:
                    best_count = count
                    best_clique = list(clique)
            if best_count == 0 or best_clique is None:
                # should not happen if remaining non-empty
                break
            cliques.append(best_clique)
            for e in list(remaining):
                if e.issubset(set(best_clique)):
                    remaining.discard(e)
        return cliques
    
    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(solve(*spec))
    return results
