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
        C_edges = []
        for (a, b) in edges:
            if a < k and b < k:
                C_edges.append((a, b))
            else:
                if a < k:
                    c, i = a, b
                else:
                    c, i = b, a
                cross_edges.append((i, c))
        
        # Data structures
        # Map i -> list of c
        i_to_c = {i: [] for i in I}
        for i, c in cross_edges:
            i_to_c[i].append(c)
            
        # Set of uncovered cross edges (i, c)
        uncovered_cross = set(cross_edges)
        
        # Set of uncovered C edges
        uncovered_C = set(frozenset(e) for e in C_edges)
        
        cliques = []
        
        # Helper to find maximal clique in a set of vertices using a graph of available edges
        def find_max_clique(vertices, available_edges):
            # vertices: list of ints
            # available_edges: set of frozensets
            if not vertices:
                return []
            
            # Greedy: sort by degree in the subgraph defined by available_edges
            def degree(v):
                return sum(1 for u in vertices if u != v and frozenset((v, u)) in available_edges)
            
            sorted_verts = sorted(vertices, key=degree, reverse=True)
            clique = []
            for v in sorted_verts:
                # Check if v is connected to all in clique
                is_connected = True
                for u in clique:
                    if frozenset((v, u)) not in available_edges:
                        is_connected = False
                        break
                if is_connected:
                    clique.append(v)
            return clique

        # Greedy loop
        while uncovered_cross:
            best_i = None
            best_S = []
            best_score = -1
            
            # Iterate through i's that have uncovered cross edges
            # To speed up, we can just pick the one with max len(uncovered neighbors)
            # But we need to check the C-edge bonus.
            
            # Optimization: Only consider i's with many uncovered neighbors
            # Let's just iterate all, it's small enough (60)
            for i in I:
                # Get uncovered neighbors for i
                uncov_neighbors = [c for (ii, c) in uncovered_cross if ii == i]
                if not uncov_neighbors:
                    continue
                
                # Find maximal clique S in uncov_neighbors using uncovered_C
                S = find_max_clique(uncov_neighbors, uncovered_C)
                
                # Calculate score
                # Cross edges covered: len(S)
                # C edges covered: count of edges in S that are in uncovered_C
                c_edges_covered = 0
                S_set = set(S)
                for c1 in S:
                    for c2 in S:
                        if c1 < c2:
                            if frozenset((c1, c2)) in uncovered_C:
                                c_edges_covered += 1
                
                score = len(S) + c_edges_covered
                
                if score > best_score:
                    best_score = score
                    best_i = i
                    best_S = S
            
            if best_i is None:
                break # Should not happen if uncovered_cross not empty
                
            # Add clique
            clique = [best_i] + best_S
            cliques.append(clique)
            
            # Update uncovered sets
            # Remove cross edges
            for c in best_S:
                uncovered_cross.discard((best_i, c))
                
            # Remove C edges
            S_set = set(best_S)
            to_remove = []
            for e in uncovered_C:
                if e.issubset(S_set):
                    to_remove.append(e)
            for e in to_remove:
                uncovered_C.discard(e)
                
        # Cover remaining C edges
        remaining_C = list(uncovered_C)
        # Convert back to tuples for processing or keep as sets
        # Use a simple greedy for pure C cliques
        # We need to cover edges in remaining_C
        # Graph is C with edges remaining_C
        
        while remaining_C:
            # Find maximal clique in the graph of remaining_C
            # Vertices involved
            verts = set()
            for e in remaining_C:
                verts.update(e)
            verts = list(verts)
            
            # Find maximal clique
            # Greedy
            def find_max_clique_remaining(vertices, edges_set):
                if not vertices:
                    return []
                def degree(v):
                    return sum(1 for u in vertices if u != v and frozenset((v, u)) in edges_set)
                sorted_verts = sorted(vertices, key=degree, reverse=True)
                clique = []
                for v in sorted_verts:
                    is_connected = True
                    for u in clique:
                        if frozenset((v, u)) not in edges_set:
                            is_connected = False
                            break
                    if is_connected:
                        clique.append(v)
                return clique

            best_clique = find_max_clique_remaining(verts, set(remaining_C))
            if not best_clique:
                # Should not happen
                break
                
            cliques.append(best_clique)
            
            # Remove covered edges
            bc_set = set(best_clique)
            new_remaining = []
            for e in remaining_C:
                if not e.issubset(bc_set):
                    new_remaining.append(e)
            remaining_C = new_remaining
            
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(solve(*spec))
    return results
