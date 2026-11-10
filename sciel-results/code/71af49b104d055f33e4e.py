def construct():
    from random import Random
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)
    
    def solve(n, k, p, seed):
        edges = instance(n, k, p, seed)
        # Build adjacency and edge sets
        C = list(range(k))
        I = list(range(k, n))
        
        # cross[i] is list of C-neighbors for i in I
        cross = {i: [] for i in I}
        # c_edges is set of frozensets for C-C edges
        c_edges = set()
        
        for (a, b) in edges:
            if a < k and b < k:
                c_edges.add(frozenset((a, b)))
            elif a >= k and b >= k:
                pass
            else:
                if a < k:
                    c, i = a, b
                else:
                    c, i = b, a
                cross[i].append(c)
        
        # 1. Create mixed cliques {i} U N(i) for all i in I
        # This covers all cross edges.
        # It also covers C-edges (u,v) if u,v are both in N(i).
        mixed_cliques = []
        covered_c_edges = set()
        
        for i in I:
            neighbors = cross[i]
            if not neighbors:
                continue # No cross edges, no clique needed? 
                # Wait, if no cross edges, i is isolated from C. 
                # But i is in I. Does i need to be in a clique? 
                # "Every edge must lie in exactly one listed clique."
                # If i has no edges, it doesn't need to be in any clique.
                # But the problem says "clique on the first k vertices...". 
                # The output is a list of cliques.
                # If i has no edges, we don't need to list a clique containing i.
                # However, the definition of the graph implies i exists.
                # If i has no edges, it contributes 0 to the edge count.
                # So we can skip it.
                pass
            else:
                clique = [i] + neighbors
                mixed_cliques.append(clique)
                # Mark C-edges within neighbors as covered
                # neighbors is a list of ints from C
                # Since C is a clique, all pairs in neighbors are edges in G.
                # We only care about covering them if they exist in c_edges.
                # But c_edges contains ALL C-edges.
                # So all pairs in neighbors are in c_edges.
                for u_idx in range(len(neighbors)):
                    u = neighbors[u_idx]
                    for v_idx in range(u_idx + 1, len(neighbors)):
                        v = neighbors[v_idx]
                        covered_c_edges.add(frozenset((u, v)))
        
        # 2. Identify remaining C-edges
        remaining_c_edges = c_edges - covered_c_edges
        
        # 3. Cover remaining C-edges with pure C-cliques
        # Build graph G_rem on C with edges remaining_c_edges
        # We need to cover these edges with cliques.
        # Since any subset of C is a clique in G, we just need to partition the edges.
        # Greedy: Find maximal cliques in G_rem, remove their edges, repeat.
        
        # Adjacency list for G_rem
        adj_rem = {u: set() for u in C}
        for e in remaining_c_edges:
            u, v = tuple(e)
            adj_rem[u].add(v)
            adj_rem[v].add(u)
            
        pure_cliques = []
        # Work on a mutable copy of remaining edges or adjacency
        # We'll remove edges as we cover them.
        # Using a set of edges is easier for removal.
        current_rem_edges = set(remaining_c_edges)
        
        while current_rem_edges:
            # Find a maximal clique in G_rem defined by current_rem_edges
            # Heuristic: Start with vertex with max degree, expand.
            
            # Get vertices involved in current_rem_edges
            active_vertices = set()
            for e in current_rem_edges:
                active_vertices.update(e)
                
            if not active_vertices:
                break
                
            # Pick starting vertex with max degree in current_rem_edges
            deg = {v: len(adj_rem[v]) for v in active_vertices}
            # Note: adj_rem is static. We need dynamic degree.
            # Let's compute degree based on current_rem_edges.
            dyn_deg = {v: 0 for v in active_vertices}
            for e in current_rem_edges:
                u, v = tuple(e)
                dyn_deg[u] += 1
                dyn_deg[v] += 1
                
            start_v = max(active_vertices, key=lambda x: dyn_deg[x])
            
            clique = [start_v]
            clique_set = {start_v}
            
            # Candidates: neighbors of start_v in current_rem_edges
            candidates = list(adj_rem[start_v] & active_vertices)
            
            # Greedy expansion
            while candidates:
                # Pick candidate with max degree in the subgraph induced by candidates + clique?
                # Or just max degree in current_rem_edges?
                # Let's pick the one with max degree in current_rem_edges restricted to clique_set?
                # Actually, to form a clique, v must be connected to ALL in clique_set.
                # We maintain candidates as those connected to all in clique_set.
                
                # Find best candidate
                # Metric: degree in current_rem_edges
                best_cand = None
                best_deg = -1
                for v in candidates:
                    # v is connected to all in clique_set by construction of candidates list update
                    d = dyn_deg[v]
                    if d > best_deg:
                        best_deg = d
                        best_cand = v
                        
                if best_cand is None:
                    break
                    
                clique.append(best_cand)
                clique_set.add(best_cand)
                
                # Update candidates: remove best_cand, and any v not connected to best_cand
                # Also, we only care about edges in current_rem_edges.
                new_candidates = []
                for v in candidates:
                    if v == best_cand:
                        continue
                    # Check if v is connected to best_cand in current_rem_edges
                    if frozenset((v, best_cand)) in current_rem_edges:
                        # Also must be connected to all other in clique_set?
                        # Yes, because v was in candidates, it was connected to all previous clique members.
                        # And best_cand is now in clique.
                        # So v is connected to all in clique_set.
                        new_candidates.append(v)
                candidates = new_candidates
                
            # We have a maximal clique 'clique'
            pure_cliques.append(clique)
            
            # Remove all edges within 'clique' from current_rem_edges
            clique_set_final = set(clique)
            edges_to_remove = []
            for e in current_rem_edges:
                if e.issubset(clique_set_final):
                    edges_to_remove.append(e)
            for e in edges_to_remove:
                current_rem_edges.discard(e)
                
        result = mixed_cliques + pure_cliques
        return result

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(solve(*spec))
    return results
