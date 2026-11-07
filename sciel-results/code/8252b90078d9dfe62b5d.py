def construct():
    from random import Random
    import sys
    sys.setrecursionlimit(10000)

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Identify Maximal Cliques
        cliques = []
        neighbors_in_C = {}
        for v in range(k, n):
            neigh = set()
            for u in range(k):
                if (u, v) in edges:
                    neigh.add(u)
            neighbors_in_C[v] = neigh
            if len(neigh) >= 1: # Clique size >= 2
                cliques.append([v] + list(neigh))
        
        C_set = set(range(k))
        C_maximal = True
        for v in range(k, n):
            if neighbors_in_C[v] == C_set:
                C_maximal = False
                break
        
        if C_maximal and k >= 2:
            cliques.append(list(range(k)))
        
        if not cliques:
            return [] # No edges? Unlikely given p=0.7 and k>=10
        
        # Precompute edges for each clique
        clique_edges = []
        for c in cliques:
            es = set()
            for i in range(len(c)):
                for j in range(i + 1, len(c)):
                    u, w = c[i], c[j]
                    e = (u, w) if u < w else (w, u)
                    es.add(e)
            clique_edges.append(es)
        
        # Map edge to list of clique indices
        edge_to_cliques = {}
        for idx, es in enumerate(clique_edges):
            for e in es:
                if e not in edge_to_cliques:
                    edge_to_cliques[e] = []
                edge_to_cliques[e].append(idx)
        
        # Verify coverage
        for e in edges:
            if e not in edge_to_cliques:
                # Fallback
                return [[u, v] for u, v in edges]
        
        # Greedy Initial Solution
        rem_greedy = set(edges)
        selected_greedy = []
        while rem_greedy:
            best_idx = -1
            max_covered = -1
            for idx, es in enumerate(clique_edges):
                # Check if any edge in es is NOT in rem_greedy?
                # No, we just count how many ARE in rem_greedy.
                # But we must ensure we don't pick a clique that has edges already covered?
                # In greedy cover, we don't care about overlap, but for partition we do.
                # However, for an initial upper bound, we can use a valid partition.
                # The seed program produces a valid partition.
                # Let's just use the seed program's logic to get a valid upper bound quickly?
                # Or just run the backtracking with a loose bound.
                
                # For now, let's just count potential coverage.
                covered = len(es.intersection(rem_greedy))
                if covered > max_covered:
                    max_covered = covered
                    best_idx = idx
            
            if max_covered == 0:
                break
                
            selected_greedy.append(best_idx)
            rem_greedy.difference_update(clique_edges[best_idx])
            
        # This greedy might produce an invalid partition (overlaps).
        # We need a valid upper bound.
        # Let's use the seed program's output as the upper bound.
        # But we don't have it here.
        # We can run a simple valid greedy:
        # Pick a clique, remove its edges. If it has edges not in rem, it's invalid.
        # So we must only pick cliques where ALL edges are in rem.
        
        rem_val = set(edges)
        valid_greedy = []
        while rem_val:
            best_idx = -1
            max_covered = -1
            for idx, es in enumerate(clique_edges):
                if es.issubset(rem_val):
                    covered = len(es)
                    if covered > max_covered:
                        max_covered = covered
                        best_idx = idx
            
            if best_idx == -1:
                # No valid clique found?
                # This means we are stuck.
                # Should not happen if maximal cliques cover all edges.
                # But maybe the remaining edges form a set that doesn't allow a full clique?
                # No, any edge is in a maximal clique.
                # If we removed edges, the remaining edges are a subset.
                # A maximal clique might not be fully contained in remaining edges.
                # We need a clique that is a subset of remaining edges.
                # This is harder.
                # Let's fallback to edge-by-edge if stuck.
                # But for upper bound, we can just use a large number.
                break
                
            valid_greedy.append(best_idx)
            rem_val.difference_update(clique_edges[best_idx])
            
        # If we didn't cover all, fallback
        if rem_val:
             # Use edge cliques
             # This is a valid partition
             return [[u, v] for u, v in edges]
             
        best_count = len(valid_greedy)
        best_solution = valid_greedy[:]
        
        # Backtracking
        def backtrack(remaining_edges, current_cliques, current_count):
            nonlocal best_count, best_solution
            
            if not remaining_edges:
                if current_count < best_count:
                    best_count = current_count
                    best_solution = current_cliques[:]
                return
            
            if current_count >= best_count:
                return
            
            # MRV
            min_candidates = float('inf')
            target_edge = None
            target_candidates = []
            
            for e in remaining_edges:
                candidates = []
                for c_idx in edge_to_cliques[e]:
                    if c_idx not in current_cliques:
                        # Check if clique c_idx is subset of remaining_edges
                        if clique_edges[c_idx].issubset(remaining_edges):
                            candidates.append(c_idx)
                
                if len(candidates) < min_candidates:
                    min_candidates = len(candidates)
                    target_edge = e
                    target_candidates = candidates
                    
                    if min_candidates == 0:
                        return
                    if min_candidates == 1:
                        break
            
            if target_edge is None:
                return

            for c_idx in target_candidates:
                current_cliques.append(c_idx)
                remaining_edges.difference_update(clique_edges[c_idx])
                
                backtrack(remaining_edges, current_cliques, current_count + 1)
                
                current_cliques.pop()
                remaining_edges.update(clique_edges[c_idx])
            
        backtrack(set(edges), [], 0)
        
        if best_count == float('inf') or not best_solution:
            return [[u, v] for u, v in edges]
        
        return [cliques[i] for i in best_solution]

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        results.append(solve_instance(n, k, p, seed))
    
    return results
