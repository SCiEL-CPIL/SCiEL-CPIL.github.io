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
        
        # Build cross adjacency and C edges
        cross = {i: set() for i in I}
        C_edges = set()
        for (a, b) in edges:
            if a < k and b < k:
                C_edges.add(frozenset((a, b)))
            elif a >= k and b >= k:
                pass
            else:
                if a < k:
                    c, i = a, b
                else:
                    c, i = b, a
                cross[i].add(c)
        
        # Identify active I vertices
        active_I = [i for i in I if len(cross[i]) > 0]
        
        # Sort active I by degree descending to prioritize large N(i)
        active_I.sort(key=lambda x: -len(cross[x]))
        
        # Set of available C-edges
        available_C_edges = set(C_edges)
        
        mixed_cliques = []
        
        # Greedy assignment of C-edges to mixed cliques
        for i in active_I:
            neighbors = cross[i]
            # We need to find a maximal clique S in the graph (neighbors, available_C_edges restricted to neighbors)
            # Graph vertices: neighbors
            # Graph edges: (u,v) in available_C_edges
            
            # Greedy maximal clique construction
            # Start with a candidate set
            candidates = list(neighbors)
            if not candidates:
                continue
            
            # Sort candidates by degree in the available graph restricted to neighbors
            # To calculate degree, we need to check availability
            def deg_in_avail(v, cand_set):
                count = 0
                for u in cand_set:
                    if u == v: continue
                    if frozenset((u, v)) in available_C_edges:
                        count += 1
                return count

            # Iterative greedy selection
            S = []
            S_set = set()
            current_candidates = list(neighbors)
            
            while current_candidates:
                # Pick vertex with max degree in current_candidates
                # Note: This is O(M^2) per step, M is size of neighbors. k is small (30).
                best_v = None
                best_deg = -1
                for v in current_candidates:
                    d = 0
                    for u in current_candidates:
                        if u == v: continue
                        if frozenset((u, v)) in available_C_edges:
                            d += 1
                    if d > best_deg:
                        best_deg = d
                        best_v = v
                
                if best_v is None:
                    break
                
                S.append(best_v)
                S_set.add(best_v)
                
                # Update candidates: must be adjacent to best_v in available graph
                # and adjacent to all other vertices in S?
                # For a clique, new vertex must be adjacent to ALL in S.
                # Since we built S greedily, we just need to check against all S.
                new_candidates = []
                for u in current_candidates:
                    if u == best_v: continue
                    is_adj_to_all = True
                    for s in S:
                        if frozenset((u, s)) not in available_C_edges:
                            is_adj_to_all = False
                            break
                    if is_adj_to_all:
                        new_candidates.append(u)
                current_candidates = new_candidates
            
            # S is the clique found
            if len(S) >= 2:
                mixed_cliques.append([i] + S)
                # Remove edges within S from available
                for idx_u in range(len(S)):
                    for idx_v in range(idx_u + 1, len(S)):
                        e = frozenset((S[idx_u], S[idx_v]))
                        available_C_edges.discard(e)
            else:
                # If S is small (0 or 1), we still need to cover cross edges.
                # The clique must contain i.
                # If S is empty, clique is [i] (invalid, size < 2).
                # But we need to cover cross edges.
                # If neighbors exist, we must include at least one neighbor?
                # If we include one neighbor c, clique [i, c] covers edge (i,c) and no C-edges.
                # This is valid.
                # If S found is empty, it means no C-edges available among neighbors?
                # Or just the greedy failed?
                # If neighbors is not empty, we should at least pick one neighbor to make a valid clique.
                # Let's pick the first neighbor.
                if neighbors:
                    c = next(iter(neighbors))
                    mixed_cliques.append([i, c])
                    # No C-edges removed
                    # Note: This doesn't cover other cross edges for i?
                    # Wait, the clique [i, c] only covers edge (i,c).
                    # Other cross edges (i, c') for c' != c are NOT covered.
                    # This is a problem.
                    # The clique MUST cover ALL cross edges incident to i.
                    # So S MUST be N(i).
                    # But S must be a clique in available_C_edges graph.
                    # If N(i) is not a clique in available_C_edges, we CANNOT use [i] U N(i).
                    
                    # This implies my previous deduction "S_i = N(i)" is WRONG if it causes conflicts.
                    # If N(i) is not a clique in the available graph, we cannot form [i] U N(i).
                    # We must split?
                    # But we can only have ONE mixed clique per i?
                    # "Every edge must lie in exactly one listed clique".
                    # If we can't cover all cross edges of i with one clique, we need MULTIPLE cliques containing i?
                    # But a clique containing i can only cover cross edges to a subset of N(i).
                    # If we use two cliques [i] U S1 and [i] U S2, the vertex i is in two cliques.
                    # Is that allowed?
                    # "Every listed set must be a clique... every edge must lie in exactly one listed clique".
                    # Vertices can be in multiple cliques. Edges cannot.
                    # So we CAN use multiple mixed cliques for the same i.
                    
                    # However, the directive says "minimize total number of cliques".
                    # Using multiple cliques for one i increases the count.
                    # Ideally, we want one.
                    # If N(i) is not a clique in the available graph, we MUST use multiple?
                    # Or we can choose S1 such that S1 is a clique, cover those, then S2 from remaining?
                    # But S2 must also be a clique in the *updated* available graph?
                    # No, the "available" graph changes as we assign edges.
                    
                    # Let's re-evaluate.
                    # If we can't fit all N(i) into one clique due to edge conflicts, we must partition N(i) into subsets S1, S2... such that each Sk is a clique in the graph of edges NOT used by previous cliques (mixed or pure).
                    
                    # Given k is small (30), and p=0.7, conflicts might be rare?
                    # But we should handle it.
                    
                    # Revised strategy for i:
                    # While there are uncovered cross-edges for i:
                    #   Find a maximal clique S in (N(i) \ already_used_for_i, available_C_edges).
                    #   Add [i] U S.
                    #   Mark edges in S as used.
                    #   Mark cross edges (i, c) for c in S as covered.
                    
                    # This might generate multiple cliques for i.
                    
                    # Let's implement this loop.
                    pass 

        # The logic above is getting complex. Let's simplify.
        # Since k is small, we can afford a slightly more expensive check.
        
        # Reset for correct implementation
        available_C_edges = set(C_edges)
        mixed_cliques = []
        covered_cross = set() # (i, c)
        
        for i in active_I:
            # Uncovered neighbors
            uncov_neigh = [c for c in cross[i] if (i, c) not in covered_cross]
            while uncov_neigh:
                # Find maximal clique in uncov_neigh using available_C_edges
                candidates = list(uncov_neigh)
                S = []
                S_set = set()
                
                while candidates:
                    best_v = None
                    best_deg = -1
                    for v in candidates:
                        d = 0
                        for u in candidates:
                            if u == v: continue
                            if frozenset((u, v)) in available_C_edges:
                                d += 1
                        if d > best_deg:
                            best_deg = d
                            best_v = v
                    if best_v is None: break
                    S.append(best_v)
                    S_set.add(best_v)
                    
                    new_cands = []
                    for u in candidates:
                        if u == best_v: continue
                        ok = True
                        for s in S:
                            if frozenset((u, s)) not in available_C_edges:
                                ok = False
                                break
                        if ok:
                            new_cands.append(u)
                    candidates = new_cands
                
                if not S:
                    # No clique of size >= 2?
                    # We must pick at least one neighbor to form a valid clique [i, c]
                    # This covers 1 cross edge and 0 C-edges.
                    c = uncov_neigh[0]
                    S = [c]
                
                # Add clique
                mixed_cliques.append([i] + S)
                
                # Update available_C_edges
                for idx_u in range(len(S)):
                    for idx_v in range(idx_u + 1, len(S)):
                        e = frozenset((S[idx_u], S[idx_v]))
                        available_C_edges.discard(e)
                        
                # Update covered_cross
                for c in S:
                    covered_cross.add((i, c))
                    
                # Update uncov_neigh
                uncov_neigh = [c for c in uncov_neigh if c not in S_set]

        # Cover remaining C-edges
        remaining_C_edges = available_C_edges
        pure_cliques = []
        
        while remaining_C_edges:
            # Find maximal clique in remaining_C_edges
            # Vertices are those incident to remaining edges
            # Greedy
            candidates = set()
            for e in remaining_C_edges:
                candidates.update(e)
            
            if not candidates: break
            
            S = []
            S_set = set()
            curr_cands = list(candidates)
            
            while curr_cands:
                best_v = None
                best_deg = -1
                for v in curr_cands:
                    d = 0
                    for u in curr_cands:
                        if u == v: continue
                        if frozenset((u, v)) in remaining_C_edges:
                            d += 1
                    if d > best_deg:
                        best_deg = d
                        best_v = v
                if best_v is None: break
                S.append(best_v)
                S_set.add(best_v)
                
                new_cands = []
                for u in curr_cands:
                    if u == best_v: continue
                    ok = True
                    for s in S:
                        if frozenset((u, s)) not in remaining_C_edges:
                            ok = False
                            break
                    if ok:
                        new_cands.append(u)
                curr_cands = new_cands
            
            if len(S) < 2:
                # Should not happen if remaining_C_edges not empty
                break
            
            pure_cliques.append(S)
            
            # Remove edges
            new_remaining = set()
            for e in remaining_C_edges:
                if not e.issubset(S_set):
                    new_remaining.add(e)
            remaining_C_edges = new_remaining

        return mixed_cliques + pure_cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(solve(*spec))
    return results
