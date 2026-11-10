def construct():
    import time
    from random import Random
    from itertools import combinations

    deadline = time.time() + 145.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        # Build cross edges
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))
        return cross_edges

    def solve_instance(n, k, p, seed, variant_seed=None):
        cross_edges = instance(n, k, p, seed)
        
        # N[v] is list of neighbors in C for v in I
        N = {v: [] for v in range(k, n)}
        for u, v in cross_edges:
            N[v].append(u)
            
        # Mixed cliques: {v} U N[v]
        mixed_cliques = []
        covered_c_edges = set()
        
        for v in range(k, n):
            if not N[v]:
                continue
            # The clique is v and all its neighbors
            clique = [int(v)] + [int(u) for u in N[v]]
            mixed_cliques.append(clique)
            
            # Mark C-edges covered by this clique
            # Note: N[v] is a subset of C, so all pairs in N[v] are edges in G
            for i in range(len(N[v])):
                for j in range(i + 1, len(N[v])):
                    u = N[v][i]
                    w = N[v][j]
                    e = (min(u, w), max(u, w))
                    covered_c_edges.add(e)
                    
        # Remaining C-edges to cover
        rem_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                e = (u, w)
                if e not in covered_c_edges:
                    rem_edges.add(e)
                    
        pure_cliques = []
        
        if not rem_edges:
            return mixed_cliques + pure_cliques
            
        # Greedy cover of rem_edges
        # We will use a heuristic to find large cliques in the remaining graph
        # Variant seed can influence tie-breaking or order
        
        rng_var = Random(variant_seed if variant_seed is not None else seed)
        
        # To improve, we can try a few different greedy strategies
        # Strategy 1: Pick vertex with max degree in rem_edges
        # Strategy 2: Pick edge with max degree sum, then expand
        
        best_pure_cliques = None
        best_count = float('inf')
        
        # Try a few random restarts or different heuristics if time permits
        # For now, let's stick to one robust heuristic per variant
        
        current_rem = set(rem_edges)
        
        while current_rem:
            if time.time() > deadline:
                # Fallback
                for e in current_rem:
                    pure_cliques.append([int(e[0]), int(e[1])])
                break
                
            # Find a maximal clique in current_rem
            # Heuristic: Start with a vertex of high degree
            # Calculate degrees in current_rem
            deg = {v: 0 for v in range(k)}
            for u, w in current_rem:
                deg[u] += 1
                deg[w] += 1
                
            # Sort vertices by degree
            vertices_sorted = sorted([v for v in deg if deg[v] > 0], key=lambda x: deg[x], reverse=True)
            
            if not vertices_sorted:
                break
                
            best_clique = []
            
            # Try starting from top few vertices to find a larger clique
            # To keep it fast, just pick the best one found in a few trials
            # Or just pick the first one.
            
            # Let's pick the vertex with max degree
            start_v = vertices_sorted[0]
            
            S = [start_v]
            S_set = {start_v}
            
            # Candidates: neighbors of start_v in current_rem
            cand = set()
            for u, w in current_rem:
                if u == start_v:
                    cand.add(w)
                elif w == start_v:
                    cand.add(u)
                    
            while cand:
                # Pick next vertex
                # Heuristic: max degree in cand (within current_rem)
                # Or max degree in current_rem?
                # Let's use degree in current_rem for stability
                best_next = None
                best_score = -1
                
                # To make it variant-sensitive, we can shuffle ties
                candidates_list = list(cand)
                if variant_seed is not None:
                    # Add small noise or shuffle
                    # Simple: just pick max, break ties randomly
                    # For now, deterministic max, but we can vary the start_v if we wanted
                    pass
                
                for v in candidates_list:
                    score = deg[v]
                    if score > best_score:
                        best_score = score
                        best_next = v
                        
                if best_next is None:
                    break
                    
                S.append(best_next)
                S_set.add(best_next)
                
                # Update candidates: must be connected to all in S
                new_cand = set()
                for v in cand:
                    if v == best_next:
                        continue
                    # Check if v is connected to best_next
                    e_check = (min(v, best_next), max(v, best_next))
                    if e_check not in current_rem:
                        continue
                    # Check if v is connected to all other in S?
                    # Since we maintain cand as intersection of neighborhoods, 
                    # v is already connected to all previous S members.
                    # We just need to intersect with neighborhood of best_next.
                    new_cand.add(v)
                    
                cand = new_cand
                
            if S:
                pure_cliques.append([int(x) for x in S])
                # Remove edges in S from current_rem
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        e = (min(S[i], S[j]), max(S[i], S[j]))
                        current_rem.discard(e)
            else:
                # Should not happen if current_rem not empty
                e = next(iter(current_rem))
                pure_cliques.append([int(e[0]), int(e[1])])
                current_rem.discard(e)
                
        return mixed_cliques + pure_cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    result = []
    
    for (n, k, p, seed) in specs:
        best_cliques = None
        best_count = float('inf')
        
        # Run base strategy
        if time.time() > deadline:
            # Fallback
            ce = instance(n, k, p, seed)
            cliques = [list(range(k))]
            for (u, v) in ce:
                cliques.append([int(u), int(v)])
            result.append(cliques)
            continue
            
        cliques = solve_instance(n, k, p, seed, None)
        count = len(cliques)
        if count < best_count:
            best_count = count
            best_cliques = cliques
            
        # Try a few variants with different random seeds for tie-breaking/shuffling
        # Although the current implementation is mostly deterministic, 
        # we can introduce randomness in the "start_v" selection or candidate ordering.
        # For now, let's just run a few times if we want to be safe, 
        # but the logic is fixed.
        
        # To potentially improve, we can try a local search on the pure clique cover.
        # But given time, let's stick to the greedy.
        
        result.append(best_cliques)

    # Ensure types
    final = []
    for partition in result:
        clean = []
        for clique in partition:
            clean.append([int(x) for x in clique])
        final.append(clean)
    return final
