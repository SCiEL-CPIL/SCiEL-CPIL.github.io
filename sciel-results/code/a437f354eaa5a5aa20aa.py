"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from collections import defaultdict

    def instance(n, k, p, seed):
        from random import Random
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

    def partition_cliques(n, k, p, seed):
        edges = instance(n, k, p, seed)
        # Build adjacency for C vertices
        adj_c = [set() for _ in range(k)]
        for (u, v) in edges:
            if u < k and v < k:
                adj_c[u].add(v)
                adj_c[v].add(u)
        
        # Neighborhoods for I vertices
        neigh_i = [set() for _ in range(n)]
        for (u, v) in edges:
            if u < k and v >= k:
                neigh_i[v].add(u)
            elif v < k and u >= k:
                neigh_i[u].add(v)
        
        cliques = []
        used_pairs = set()
        
        # Process I vertices
        # We want to partition N(v) into subsets S such that pairs in S are not used by other subsets
        for v in range(k, n):
            N_v = list(neigh_i[v])
            if not N_v:
                continue
            
            # Greedily form subsets of N_v
            remaining = set(N_v)
            while remaining:
                # Pick a start vertex
                start = min(remaining)
                current_subset = {start}
                remaining.remove(start)
                
                # Try to add more vertices to current_subset
                # We can add w if for all x in current_subset, (x, w) is not in used_pairs
                # And we should prefer to add vertices that allow larger subsets
                # Simple greedy: iterate through remaining, add if possible
                added = True
                while added:
                    added = False
                    for w in list(remaining):
                        can_add = True
                        for x in current_subset:
                            pair = (min(x, w), max(x, w))
                            if pair in used_pairs:
                                can_add = False
                                break
                        if can_add:
                            current_subset.add(w)
                            remaining.remove(w)
                            added = True
                            # Mark pairs within current_subset as used?
                            # We should mark them when the subset is finalized to be safe,
                            # but marking as we go prevents adding conflicting vertices.
                            # However, if we mark now, and then fail to finalize? 
                            # We always finalize the current_subset.
                            # So we can mark pairs within current_subset as used immediately?
                            # No, because if we add w, we use (x,w) for x in current_subset.
                            # These pairs are "consumed" by this clique.
                            # So we should add them to used_pairs.
                            for x in current_subset:
                                if x != w:
                                    pair = (min(x, w), max(x, w))
                                    used_pairs.add(pair)
                
                # Finalize clique
                if len(current_subset) >= 2:
                    cliques.append([v] + sorted(current_subset))
                else:
                    # Single vertex, no C-C edges covered
                    cliques.append([v, start])
        
        # Cover remaining C-C edges
        # Remaining edges are those in C not in used_pairs
        remaining_edges = []
        for u in range(k):
            for w in range(u + 1, k):
                if (u, w) not in used_pairs:
                    remaining_edges.append((u, w))
        
        # Greedy clique cover for remaining_edges
        # Build graph of remaining edges
        rem_adj = [set() for _ in range(k)]
        for (u, w) in remaining_edges:
            rem_adj[u].add(w)
            rem_adj[w].add(u)
        
        covered = [False] * (k * k) # or just use a set of remaining edges
        rem_edge_set = set(remaining_edges)
        
        while rem_edge_set:
            # Find a maximal clique in the graph induced by rem_edge_set
            # Start with a vertex with high degree in rem graph
            # Simple: pick first edge, try to extend
            first_edge = next(iter(rem_edge_set))
            u, w = first_edge
            clique = {u, w}
            
            # Try to add vertices
            candidates = [x for x in range(k) if x not in clique]
            added = True
            while added:
                added = False
                for x in candidates:
                    # Check if x is connected to all in clique
                    is_connected = all((min(x, y), max(x, y)) in rem_edge_set for y in clique)
                    if is_connected:
                        clique.add(x)
                        candidates.remove(x)
                        added = True
            
            # Add clique
            cliques.append(sorted(clique))
            
            # Remove edges in this clique from rem_edge_set
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    pair = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                    if pair in rem_edge_set:
                        rem_edge_set.remove(pair)
        
        return cliques

    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe
    
    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            break
        results.append(partition_cliques(*spec))
    
    return results
# EVOLVE-BLOCK-END
