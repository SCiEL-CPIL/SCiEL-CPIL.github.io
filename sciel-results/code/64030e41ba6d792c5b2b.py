"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from random import Random

    deadline = time.time() + 140.0  # safety margin before the 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        # Build the exact same split graph as the seed generator.
        # Clique on first k vertices, independent set on the rest.
        # Cross pairs joined with probability p.
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))
        return cross_edges

    def solve_instance(n, k, p, seed):
        cross_edges = instance(n, k, p, seed)
        
        # N[v] is list of neighbors in C for v in I
        N = {v: [] for v in range(k, n)}
        for u, v in cross_edges:
            N[v].append(u)
            
        # Available C-edges (edges within the clique part that need to be covered)
        # Using a set of tuples (min, max) for O(1) lookup
        available = set()
        for u in range(k):
            for w in range(u + 1, k):
                available.add((u, w))
                
        mixed_cliques = []
        
        # Sort I vertices by degree descending to handle "harder" ones first
        I_vertices = sorted(range(k, n), key=lambda x: len(N[x]), reverse=True)
        
        for v in I_vertices:
            if not N[v]:
                continue
                
            remaining_neighbors = set(N[v])
            
            while remaining_neighbors:
                if time.time() > deadline:
                    # Fallback: just add 2-cliques for remaining
                    for u in remaining_neighbors:
                        mixed_cliques.append([int(v), int(u)])
                    break
                
                # Find a maximal clique in the graph induced by remaining_neighbors on 'available' edges.
                # We want a subset S of remaining_neighbors such that all edges in S are in available.
                # This allows us to form a clique [v] + S, which covers cross-edges (v, u) for u in S
                # AND the clique-edges (u, w) for u, w in S.
                
                # Greedy maximal clique finder on the subgraph defined by 'available' restricted to remaining_neighbors
                S = []
                S_set = set()
                cand_set = set(remaining_neighbors)
                
                while cand_set:
                    if time.time() > deadline:
                        # Fallback
                        for u in remaining_neighbors:
                            mixed_cliques.append([int(v), int(u)])
                        remaining_neighbors = set()
                        break
                    
                    # Find vertex with max degree in cand_set within the 'available' graph
                    best_v = None
                    best_deg = -1
                    for w in cand_set:
                        deg = 0
                        for x in cand_set:
                            if x == w: continue
                            e = (min(w, x), max(w, x))
                            if e in available:
                                deg += 1
                        if deg > best_deg:
                            best_deg = deg
                            best_v = w
                            
                    if best_v is None:
                        break
                        
                    S.append(best_v)
                    S_set.add(best_v)
                    
                    # Update cand_set: keep only neighbors of best_v in cand_set
                    new_cand = set()
                    for w in cand_set:
                        if w == best_v: continue
                        e = (min(best_v, w), max(best_v, w))
                        if e in available:
                            new_cand.add(w)
                    cand_set = new_cand
                
                if S:
                    # The clique is [v] + S
                    # This covers cross-edges (v, u) for all u in S
                    # And clique-edges (u, w) for all u, w in S
                    mixed_cliques.append([int(v)] + [int(u) for u in S])
                    
                    # Remove the clique-edges from available
                    for i in range(len(S)):
                        for j in range(i + 1, len(S)):
                            e = (min(S[i], S[j]), max(S[i], S[j]))
                            available.discard(e)
                    
                    # Remove S from remaining_neighbors
                    remaining_neighbors.difference_update(S_set)
                else:
                    # No clique of size > 1 can be formed from remaining_neighbors using available edges?
                    # This means no two vertices in remaining_neighbors are connected in 'available'.
                    # So we must cover them as 2-cliques (v, u).
                    # But wait, if S is empty, it means we couldn't pick any vertex?
                    # The loop picks best_v if cand_set is not empty.
                    # So S will have at least 1.
                    # If S has size 1, say [u], then we add [v, u].
                    # This covers cross-edge (v, u).
                    # It does NOT cover any clique-edge because there is only one vertex in S.
                    # So this is valid.
                    pass

        # Now cover remaining available edges with pure cliques (within C)
        pure_cliques = []
        rem_edges = set(available)
        
        while rem_edges:
            if time.time() > deadline:
                # Fallback: add edges as 2-cliques
                for e in rem_edges:
                    pure_cliques.append([int(e[0]), int(e[1])])
                break
                
            # Find a maximal clique in the graph defined by rem_edges
            # Vertices involved in rem_edges
            vertices = set()
            for u, w in rem_edges:
                vertices.add(u)
                vertices.add(w)
                
            # Greedy maximal clique on rem_edges
            S = []
            S_set = set()
            cand_set = set(vertices)
            
            while cand_set:
                if time.time() > deadline:
                    for e in rem_edges:
                        pure_cliques.append([int(e[0]), int(e[1])])
                    rem_edges = set()
                    break
                
                # Find vertex with max degree in cand_set within rem_edges
                best_v = None
                best_deg = -1
                for w in cand_set:
                    deg = 0
                    for x in cand_set:
                        if x == w: continue
                        e = (min(w, x), max(w, x))
                        if e in rem_edges:
                            deg += 1
                    if deg > best_deg:
                        best_deg = deg
                        best_v = w
                        
                if best_v is None:
                    break
                    
                S.append(best_v)
                S_set.add(best_v)
                
                # Update cand_set: keep only neighbors of best_v in cand_set
                new_cand = set()
                for w in cand_set:
                    if w == best_v: continue
                    e = (min(best_v, w), max(best_v, w))
                    if e in rem_edges:
                        new_cand.add(w)
                cand_set = new_cand
                
            if S:
                pure_cliques.append([int(u) for u in S])
                # Remove edges in S from rem_edges
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        e = (min(S[i], S[j]), max(S[i], S[j]))
                        rem_edges.discard(e)
            else:
                # Should not happen if rem_edges is not empty
                break

        # Combine: one big clique on C is NOT used. Instead, we use pure_cliques and mixed_cliques.
        # Wait, the current program uses one big clique [0..k-1] to cover all C-edges.
        # The inspiration uses a greedy approach to cover C-edges with mixed cliques and remaining pure cliques.
        # The inspiration's approach is more complex but potentially better.
        # Let's check if the inspiration's approach is valid.
        # Yes, every edge is covered exactly once.
        
        return mixed_cliques + pure_cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    result = []
    for (n, k, p, seed) in specs:
        if time.time() > deadline:
            # Fallback: just use the big clique + all cross edges as 2-cliques
            ce = instance(n, k, p, seed)
            cliques = [list(range(k))]
            for (u, v) in ce:
                cliques.append([int(u), int(v)])
            result.append(cliques)
        else:
            result.append(solve_instance(n, k, p, seed))

    # Ensure all types are Python int and lists
    final = []
    for partition in result:
        clean = []
        for clique in partition:
            clean.append([int(x) for x in clique])
        final.append(clean)
    return final
# EVOLVE-BLOCK-END
