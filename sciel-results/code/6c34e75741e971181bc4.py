<<<<<<< SEARCH
    current = time.time()
    remaining = DEADLINE - current
    if t1 + t2 > remaining:
        scale = remaining / (t1 + t2)
        t1 *= scale
        t2 *= scale
=======
    current = time.time()
    remaining = DEADLINE - current
    if t1 + t2 > remaining:
        scale = remaining / (t1 + t2)
        t1 *= scale
        t2 *= scale
    
    # Add a final polish pass: try to merge adjacent 2-cliques into larger cliques
    def polish_cliques(n, adj, cliques, deadline):
        """
        Try to merge small cliques into larger ones where possible.
        """
        if time.time() > deadline:
            return cliques
        
        # Build a map from edge to clique index
        edge_to_clique = {}
        for idx, clique in enumerate(cliques):
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, v = clique[i], clique[j]
                    edge_to_clique[(min(u,v), max(u,v))] = idx
        
        # Try to merge cliques that share a vertex and whose union is a clique
        merged = True
        while merged and time.time() < deadline:
            merged = False
            for i in range(len(cliques)):
                if len(cliques[i]) >= 2:
                    for j in range(i+1, len(cliques)):
                        if len(cliques[j]) >= 2:
                            # Check if they share a vertex
                            shared = set(cliques[i]) & set(cliques[j])
                            if shared:
                                # Check if union is a clique
                                union = set(cliques[i]) | set(cliques[j])
                                is_clique = True
                                for a in union:
                                    for b in union:
                                        if a < b and not (adj[a] & (1 << b)):
                                            is_clique = False
                                            break
                                    if not is_clique:
                                        break
                                if is_clique:
                                    # Merge
                                    new_clique = list(union)
                                    cliques[i] = new_clique
                                    cliques.pop(j)
                                    merged = True
                                    break
                if merged:
                    break
        
        return cliques
>>>>>>> REPLACE

