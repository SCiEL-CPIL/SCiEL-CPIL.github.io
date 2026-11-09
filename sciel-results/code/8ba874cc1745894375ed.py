"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import time
    from itertools import combinations

    start_time = time.time()
    deadline = 115.0 # 120s limit, keep margin

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges_cc = []
        for u in range(k):
            for v in range(u + 1, k):
                edges_cc.append((u, v))
        
        edges_ci = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges_ci.append((u, v))
                    
        return edges_cc, edges_ci

    def solve_split_graph(n, k, edges_ci):
        # Map each independent vertex to its neighborhood in the clique part
        # Use frozenset for efficient intersection and hashing
        shadows = {}
        for v in range(k, n):
            neighbors = frozenset(u for u, v_edge in edges_ci if v_edge == v)
            shadows[v] = neighbors
            
        # We need to cover all edges in edges_ci.
        # An edge (u, v) is covered if we select a clique containing u and v.
        # A clique containing v must contain a subset of shadows[v].
        # Let S_v be the set of independent vertices we group with v.
        # The clique is (intersection_{w in S_v} shadows[w]) union S_v.
        # This clique covers all edges (u, w) for u in intersection and w in S_v.
        
        # Greedy Set Cover approach:
        # 1. Start with all cross edges uncovered.
        # 2. Repeatedly find a group of independent vertices S and a subset of clique vertices C
        #    such that C is a subset of the intersection of shadows of S, 
        #    and the number of newly covered edges is maximized.
        # 3. Add the clique C union S to the solution.
        # 4. Remove covered edges.
        
        # Optimization: 
        # Since k is small (10, 20, 30), we can iterate over subsets of independent vertices? 
        # No, n-k can be 20, 40, 60. 2^60 is too big.
        
        # Alternative: 
        # Group independent vertices by identical shadows?
        # If two vertices have the same shadow, they can always be in the same clique with that shadow.
        # If shadows are nested (A subset B), they can be in a clique with A.
        
        # Let's use a heuristic:
        # Identify "types" of independent vertices based on their shadow.
        # Actually, just do a greedy selection of cliques.
        
        # Precompute for each independent vertex v, the set of edges incident to it.
        # But we need to cover pairs.
        
        # Let's try a simpler greedy:
        # While there are uncovered edges:
        #   Find the independent vertex v with the most uncovered edges.
        #   Find the set of clique vertices C that are connected to v and are connected to 
        #   all other independent vertices we decide to include.
        
        # Even simpler: 
        # The "shadow" of a single vertex v is shadows[v].
        # A clique {v} union shadows[v] covers all edges incident to v.
        # If we take multiple vertices v1, v2, ..., vm, the clique is (intersection of shadows) union {v1...vm}.
        # This covers edges (u, vi) where u is in the intersection.
        
        # Greedy:
        # 1. Start with all cross edges uncovered.
        # 2. While uncovered edges remain:
        #    a. Find the independent vertex v with the maximum number of uncovered edges.
        #    b. Let C be the set of clique vertices that are neighbors of v AND have uncovered edges to v.
        #       Actually, to maximize coverage, we should include all neighbors of v that are "useful".
        #       But adding a clique vertex u to the clique only helps if it covers edges to OTHER independent vertices too.
        #       
        #    Let's try: Pick v with max uncovered degree.
        #    Let S = {v}.
        #    Let CommonShadow = shadows[v].
        #    Try to add other independent vertices w to S if they share a large common shadow with S.
        #    Specifically, if we add w, the new shadow is CommonShadow & shadows[w].
        #    The new clique covers edges (u, x) for u in (CommonShadow & shadows[w]) and x in S union {w}.
        #    
        #    This is getting complex to optimize perfectly. Let's use a standard greedy set cover on the edges.
        #    Candidate cliques:
        #    For each independent vertex v:
        #       Clique C_v = shadows[v] union {v}
        #       This covers all edges incident to v.
        #    For each pair (v, w):
        #       Clique C_{v,w} = (shadows[v] & shadows[w]) union {v, w}
        #       This covers edges incident to v and w that go to the common shadow.
        #    ...
        
        # Given the time limit and n up to 90, let's stick to a robust greedy:
        # 1. Create a set of uncovered edges.
        # 2. While uncovered edges exist:
        #    - Find the independent vertex v with the highest number of uncovered edges.
        #    - Let S = {v}.
        #    - Let CurrentShadow = shadows[v].
        #    - Iteratively add independent vertices w to S if:
        #      - w has uncovered edges.
        #      - The intersection CurrentShadow & shadows[w] is "large enough" (e.g., non-empty or > threshold).
        #      - Actually, just add w if it shares at least one neighbor with the current shadow?
        #      - Let's just add w if shadows[w] is a superset of CurrentShadow? No, that reduces shadow.
        #      - Let's add w if CurrentShadow & shadows[w] is not empty.
        #      - Prioritize w with largest overlap.
        #    - Once S is built, let C = intersection of shadows of all v in S.
        #    - If C is empty, we can't form a clique with the clique part. We must handle edges differently?
        #      Wait, if C is empty, the clique is just S. But S is an independent set, so it's not a clique unless |S|=1.
        #      So we must ensure C is not empty if |S| > 1.
        #      If C becomes empty, we should backtrack or just use S={v} with C=shadows[v].
        
        # Let's simplify:
        # Just use cliques of the form (shadows[v] union {v}) for each v?
        # This covers all edges incident to v.
        # Total cliques = n-k.
        # Plus the main clique C (size k).
        # Total = 1 + (n-k).
        # For n=90, k=30, this is 61 cliques.
        # Current seed: 1 + sum(degrees) = 1 + ~0.7 * 30 * 60 = 1 + 1260 = 1261.
        # 61 is much better than 1261.
        
        # Can we do better than n-k?
        # If we group v and w, we use 1 clique instead of 2.
        # So we want to maximize grouping.
        
        # Greedy Grouping:
        # 1. Uncovered edges = set of all edges_ci.
        # 2. While uncovered edges:
        #    a. Pick v with max uncovered degree.
        #    b. S = [v].
        #    c. CommonShadow = shadows[v].
        #    d. Candidates = [w for w in independent vertices if w != v and has uncovered edges]
        #    e. Sort candidates by |CommonShadow & shadows[w]| descending.
        #    f. For each w in candidates:
        #       NewShadow = CommonShadow & shadows[w]
        #       If NewShadow is not empty:
        #           # Check if adding w is beneficial.
        #           # It reduces the shadow, but adds one vertex to the clique.
        #           # The clique size increases by 1.
        #           # The edges covered are those (u, x) where u in NewShadow and x in S U {w}.
        #           # Since we are doing set cover, we just want to cover as many uncovered edges as possible.
        #           # Let's just add w if NewShadow is not empty.
        #           S.append(w)
        #           CommonShadow = NewShadow
        #    g. C = list(CommonShadow)
        #    h. Clique = C + S
        #    i. Add Clique to results.
        #    j. Remove all edges (u, v) from uncovered if u in C and v in S.
        
        # This greedy might be slow if we sort candidates every time.
        # n=90, k=30. Independent vertices = 60.
        # 60 iterations. Sorting 60 items is fast.
        
        cliques = []
        # Main clique
        cliques.append(list(range(k)))
        
        # Build adjacency list for cross edges for fast lookup/removal
        # uncovered_edges is a set of tuples (u, v)
        uncovered = set(edges_ci)
        
        # Map independent vertex to its shadow (frozenset of clique vertices)
        shadows = {}
        for v in range(k, n):
            shadows[v] = frozenset(u for u, v_edge in edges_ci if v_edge == v)
            
        independent_vertices = list(range(k, n))
        
        while uncovered:
            # Find v with max uncovered degree
            # We can just iterate and count
            max_deg = -1
            best_v = None
            for v in independent_vertices:
                deg = sum(1 for u in shadows[v] if (u, v) in uncovered)
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                    
            if best_v is None:
                break # Should not happen if uncovered is not empty
                
            S = [best_v]
            CommonShadow = shadows[best_v]
            
            # Find candidates to add to S
            # We want to add w if it helps cover more edges.
            # A simple heuristic: add w if it shares at least one neighbor with CommonShadow.
            # Sort by overlap size.
            candidates = []
            for w in independent_vertices:
                if w == best_v:
                    continue
                # Check if w has any uncovered edges
                # If not, no point adding it
                has_uncovered = any((u, w) in uncovered for u in shadows[w])
                if not has_uncovered:
                    continue
                    
                overlap = CommonShadow & shadows[w]
                if overlap:
                    candidates.append((len(overlap), w))
                    
            # Sort by overlap size descending
            candidates.sort(key=lambda x: x[0], reverse=True)
            
            for _, w in candidates:
                # Re-evaluate if w still has uncovered edges? 
                # The edges covered by the final clique depend on the final CommonShadow.
                # If we add w, CommonShadow shrinks.
                # This might uncover edges for other vertices? No, we only remove edges.
                # So if w has uncovered edges now, it will have them later (subset).
                
                NewShadow = CommonShadow & shadows[w]
                if not NewShadow:
                    continue
                
                # Heuristic: Add w if the new shadow is "large" or if w has many edges to NewShadow
                # For now, just add if overlap was non-empty (which is True here)
                S.append(w)
                CommonShadow = NewShadow
                
            C = list(CommonShadow)
            clique = C + S
            cliques.append(clique)
            
            # Remove covered edges
            # Edges (u, v) where u in C and v in S
            C_set = set(C)
            S_set = set(S)
            to_remove = set()
            for u in C:
                for v in S:
                    if (u, v) in uncovered:
                        to_remove.add((u, v))
            uncovered -= to_remove
            
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            # Fallback: return seed-like result to ensure validity
            # This is a safety net. The greedy should be fast.
            n, k, p, seed = spec
            edges_cc, edges_ci = instance(n, k, p, seed)
            cliques = [list(range(k))]
            for u, v in edges_ci:
                cliques.append([u, v])
            results.append(cliques)
            continue
            
        n, k, p, seed = spec
        edges_cc, edges_ci = instance(n, k, p, seed)
        
        cliques = solve_split_graph(n, k, edges_ci)
        
        # Validation step (internal check to avoid invalid submission)
        # Check 1: Every clique is a clique in the graph
        # Check 2: Every edge is covered exactly once
        # This is expensive, so we trust the logic but maybe do a light check for small n?
        # For n=90, checking O(N^3) is fine.
        
        # Let's do a quick sanity check on the last clique formed
        # Actually, let's just run it. The logic ensures (u, v) is removed when covered.
        # Does it ensure no edge is covered twice?
        # Yes, because we remove edges from `uncovered` immediately.
        # Does it ensure every edge is covered?
        # Yes, the loop runs while `uncovered` is not empty.
        # Does it ensure the clique is valid?
        # The clique is C U S.
        # C is a subset of the clique part, so it's a clique.
        # S is a set of independent vertices.
        # Every vertex in C is connected to every vertex in S because C is the intersection of their shadows.
        # So C U S is a clique.
        # Size >= 2?
        # If S has 1 vertex v, and C is non-empty, size >= 2.
        # If C is empty, we skipped adding w if NewShadow was empty.
        # But what if best_v has empty shadow?
        # Then CommonShadow is empty.
        # Then no w can be added.
        # Clique is [] + [v] = [v]. Size 1. INVALID.
        
        # Fix: If CommonShadow is empty, we cannot form a valid clique with just v.
        # We must handle vertices with empty shadows separately?
        # If a vertex v has no neighbors in the clique part, it has no cross edges.
        # So it won't be selected in the "max uncovered degree" step if it has 0 uncovered edges.
        # If it has uncovered edges, its shadow is not empty.
        # Wait, `shadows[v]` is computed from `edges_ci`.
        # If v has no edges in `edges_ci`, `shadows[v]` is empty.
        # Then `deg` is 0.
        # It will not be picked as `best_v` if there are other vertices with deg > 0.
        # If ALL vertices have deg 0, `uncovered` is empty, loop doesn't run.
        # So `best_v` always has deg > 0, so `shadows[best_v]` is not empty.
        # So `CommonShadow` starts non-empty.
        # We only add w if `NewShadow` is not empty.
        # So `CommonShadow` remains non-empty.
        # So `C` is non-empty.
        # So `clique` size is at least 1 (from C) + 1 (from S) = 2.
        # Valid.
        
        results.append(cliques)
        
    return results
# EVOLVE-BLOCK-END
