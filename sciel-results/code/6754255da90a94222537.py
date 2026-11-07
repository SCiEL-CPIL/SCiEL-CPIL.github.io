def construct():
    from random import Random
    import time

    start_time = time.time()
    DEATHLINE = 138.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set((min(u, v), max(u, v)) for u, v in edges)
        
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        C = list(range(k))
        I = list(range(k, n))
        
        # Pre-compute cross neighborhoods
        # For each v in I, N_C[v] = {u in C : (u,v) in edge_set}
        N_C = {v: adj[v] & set(C) for v in I}
        # For each u in C, N_I[u] = {v in I : (u,v) in edge_set}
        N_I = {u: adj[u] & set(I) for u in C}
        
        cliques = []
        uncovered = set(edge_set)
        
        # Helper to check if a set of vertices forms a clique and count uncovered edges
        def try_add_clique(verts):
            if len(verts) < 2:
                return 0
            # Check clique property
            vs = list(verts)
            for i in range(len(vs)):
                for j in range(i + 1, len(vs)):
                    if vs[j] not in adj[vs[i]]:
                        return 0
            # Count uncovered edges
            count = 0
            for i in range(len(vs)):
                for j in range(i + 1, len(vs)):
                    e = (min(vs[i], vs[j]), max(vs[i], vs[j]))
                    if e in uncovered:
                        count += 1
            return count
        
        def add_clique(verts):
            vs = sorted(verts)
            cliques.append(vs)
            for i in range(len(vs)):
                for j in range(i + 1, len(vs)):
                    e = (min(vs[i], vs[j]), max(vs[i], vs[j]))
                    uncovered.discard(e)
        
        # Phase 1: Add the full clique C
        if k >= 2:
            add_clique(C)
        
        # Phase 2: Greedy set cover on remaining edges
        # We want to cover cross edges (u,v) where u in C, v in I
        # A clique containing v and a subset of N_C[v] covers all edges (u,v) for u in that subset
        # To maximize efficiency, we want to find large cliques.
        # In a split graph, any clique is either subset of C, or {v} U S where v in I and S subset of N_C[v].
        # Since C is already covered, we only need to cover cross edges.
        # A clique {v} U S covers edges (u,v) for u in S.
        # Note: It also covers edges within S, but those are already covered by C.
        # So the "value" of clique {v} U S is |S| (number of new cross edges covered).
        # Wait, is it possible that a clique covers edges between two I vertices? No, I is independent.
        # So every new edge covered must be a cross edge.
        
        # Strategy: For each v in I, the best clique involving v is {v} U N_C[v] (all its neighbors in C).
        # This covers |N_C[v]| edges.
        # However, we might not want to cover ALL edges incident to v if it prevents better grouping?
        # No, because edges are distinct. Covering an edge is always good.
        # But we pay 1 clique for each v we use.
        # So we should only use v if it covers at least 1 edge.
        # Actually, we can just iterate through v in I and if it has uncovered edges, add {v} U {uncovered neighbors in C}.
        
        # Let's refine:
        # For each v in I, let U_v = {u in C : (u,v) in uncovered}
        # If U_v is not empty, we can form clique {v} U U_v.
        # This covers all edges in U_v incident to v.
        # Is this optimal?
        # Suppose we have v1 and v2.
        # Option A: Clique {v1} U U_v1, Clique {v2} U U_v2. Cost 2.
        # Option B: Can we merge? No, v1 and v2 are not adjacent. So they can't be in the same clique.
        # So each v in I must be in its own clique (or not used if no edges).
        # Wait, can a clique contain multiple I vertices? No, I is independent.
        # So yes, each clique covering cross edges must contain exactly one vertex from I.
        # Therefore, the problem reduces to: for each v in I, cover its incident uncovered edges.
        # The most efficient way is to put all its uncovered neighbors in C into the clique with it.
        # Because any subset of N_C[v] forms a clique with v.
        # So we just iterate through I, and for each v, if it has uncovered edges to C, add {v} U {those neighbors}.
        
        # This is O(n) cliques max.
        # Let's check if we can do better by NOT covering all edges of v at once?
        # No, covering more edges per clique is always better (or equal) since the cost is per clique.
        # So the greedy strategy of "for each v, take all its remaining neighbors" is actually optimal for split graphs?
        # Let's verify.
        # Total cliques = 1 (for C) + number of v in I with degree > 0 to C.
        # Is it possible to reduce this?
        # The edges (u,v) must be covered.
        # Each clique can cover edges for at most one v in I.
        # So we need at least max_v (deg(v) / max_clique_size?) No.
        # We need at least the number of v's that have any edge, because a clique can't cover edges for two different v's.
        # So the lower bound is |{v in I : deg(v) > 0}|.
        # And our algorithm achieves exactly that (plus the 1 for C).
        # So for split graphs, this is optimal!
        
        # Wait, did I miss something?
        # "Minimize the total number of cliques"
        # If I have v1 connected to u1, u2.
        # Clique {v1, u1, u2} covers (v1,u1), (v1,u2).
        # If I have v2 connected to u1, u2.
        # Clique {v2, u1, u2} covers (v2,u1), (v2,u2).
        # Total 2 cliques for I + 1 for C = 3.
        # Can I do better?
        # No, v1 and v2 can't be in same clique.
        # So yes, it seems optimal for split graphs.
        
        # Let's double check if the "Clique" definition allows using vertices from C that are not neighbors?
        # No, it must be a clique. So only neighbors.
        
        # So the algorithm is:
        # 1. Add C.
        # 2. For each v in I, if v has uncovered edges to C, add {v} U {uncovered neighbors of v in C}.
        
        # This is extremely fast.
        
        for v in I:
            if time.time() > DEATHLINE:
                break
            # Find uncovered neighbors in C
            unc_neigh = []
            for u in C:
                if (min(u, v), max(u, v)) in uncovered:
                    unc_neigh.append(u)
            if unc_neigh:
                add_clique([v] + unc_neigh)
        
        # Any remaining edges?
        # Since I is independent and C is a clique, all edges are either in C or cross.
        # We covered all edges in C with clique C.
        # We covered all cross edges by iterating over I.
        # So uncovered should be empty.
        # Just in case, add 2-cliques for any leftovers.
        for e in sorted(uncovered):
            cliques.append(list(e))
            
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        if time.time() > DEATHLINE - 2:
            # Fallback to simple edge listing if we run out of time
            edges = instance(n, k, p, seed)
            results.append([list(e) for e in edges])
            continue
            
        cliques = solve_instance(n, k, p, seed)
        results.append(cliques)
    
    final = []
    for clique_list in results:
        final.append([list(int(v) for v in clique) for clique in clique_list])
    
    return final
