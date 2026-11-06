<<<<<<< SEARCH
    def perturb_and_refine(edges_set, seed_val, num_remove):
        """
        Remove some edges randomly, then run greedy + local search to refine.
        """
        random.seed(seed_val)
        edges = set(edges_set)
        
        # Remove num_remove random edges
        to_remove = random.sample(list(edges), min(num_remove, len(edges)))
        for e in to_remove:
            edges.remove(e)
        
        # Rebuild adj
        adj = build_adj(edges)
        
        # Run greedy
        possible_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    possible_edges.append((i, j))
        
        random.shuffle(possible_edges)
        
        for u, v in possible_edges:
            if time.time() > deadline - 5:
                break
            if not would_create_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.add((u, v))
        
        # Run local search
        edges = local_search(edges, seed_val)
        
        return edges
=======
    def perturb_and_refine(edges_set, seed_val, num_remove):
        """
        Remove some edges randomly, then run greedy + local search to refine.
        """
        random.seed(seed_val)
        edges = set(edges_set)
        
        # Remove num_remove random edges
        to_remove = random.sample(list(edges), min(num_remove, len(edges)))
        for e in to_remove:
            edges.remove(e)
        
        # Rebuild adj
        adj = build_adj(edges)
        
        # Run greedy
        possible_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    possible_edges.append((i, j))
        
        random.shuffle(possible_edges)
        
        for u, v in possible_edges:
            if time.time() > deadline - 5:
                break
            path3 = compute_path3_adj(adj)
            if not would_create_c6(adj, u, v, path3):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.add((u, v))
        
        # Run local search
        edges = local_search(edges, seed_val)
        
        return edges
>>>>>>> REPLACE
