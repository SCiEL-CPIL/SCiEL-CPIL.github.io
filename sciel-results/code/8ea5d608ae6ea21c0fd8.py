<<<<<<< SEARCH
    def solve_instance(n, p, seed):
        """Solve one instance: find minimum transversal."""
        edges = generate_graph(n, p, seed)
        adj = build_adjacency(n, edges)
        
        # Get all maximal cliques of size >= 2
        cliques = bron_kerbosch_maximal_cliques(adj, n)
        cliques = [c for c in cliques if len(c) >= 2]
        
        if not cliques:
            return []
        
        # Greedy set cover
        selected = greedy_transversal(cliques, n)
        
        # Local search improvement
        remaining_time = deadline - time.time()
        if remaining_time > 5:
            selected = local_search_improve(selected, cliques, n, max_iters=10000)
        
        return sorted(selected)
=======
    def solve_instance(n, p, seed):
        """Solve one instance: find minimum transversal."""
        edges = generate_graph(n, p, seed)
        adj = build_adjacency(n, edges)
        
        # Get all maximal cliques of size >= 2
        cliques = bron_kerbosch_maximal_cliques(adj, n)
        cliques = [c for c in cliques if len(c) >= 2]
        
        if not cliques:
            return []
        
        # Multi-start greedy + local search, keep best
        best_selected = None
        best_size = float('inf')
        
        # Deterministic greedy first
        sel = greedy_transversal_bitmask(cliques, n, rng=None)
        remaining_time = deadline - time.time()
        if remaining_time > 5:
            sel = local_search_improve_fast(sel, cliques, n, deadline=deadline)
        if len(sel) < best_size:
            best_size = len(sel)
            best_selected = set(sel)
        
        # Randomized multi-start if time permits
        rng = random.Random(seed + 9999)
        start = time.time()
        while time.time() < deadline - 10:
            sel = greedy_transversal_bitmask(cliques, n, rng=rng, alpha=0.5)
            remaining_time = deadline - time.time()
            if remaining_time > 5:
                sel = local_search_improve_fast(sel, cliques, n, deadline=deadline)
            if len(sel) < best_size:
                best_size = len(sel)
                best_selected = set(sel)
            # If we've hit a good size, do a few more then stop
            if best_size <= 8 and time.time() - start > 3:
                break
        
        return sorted(best_selected) if best_selected else []
>>>>>>> REPLACE
