"""Program: minimize total number of cliques in an edge-exact clique partition of two dense random graphs."""
# EVOLVE-BLOCK-START
def construct():
    def random_graph(n, p, seed):
        import random
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def partition_cliques(edges):
        if not edges:
            return []
        
        n = max(max(e) for e in edges) + 1 if edges else 0
        # adj_mask[v] is a bitmask of neighbors in the UNCOVERED graph
        adj_mask = [0] * n
        
        # Initialize masks
        for u, v in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            
        uncovered = set(edges)
        cliques = []
        
        while uncovered:
            # Find the "densest" edge (u, v) to start the clique.
            # Density is defined as the number of common neighbors in the uncovered graph.
            best_u = -1
            best_v = -1
            max_common = -1
            
            # Iterate through all uncovered edges to find the one with max common neighbors
            for u, v in uncovered:
                # Common neighbors are bits set in both adj_mask[u] and adj_mask[v]
                common = adj_mask[u] & adj_mask[v]
                # Count bits
                c_count = common.bit_count()
                if c_count > max_common:
                    max_common = c_count
                    best_u = u
                    best_v = v
                    
            if best_u == -1:
                # Fallback if something goes wrong, take any edge
                e = next(iter(uncovered))
                best_u, best_v = e
                
            # Expand (best_u, best_v) to a maximal clique
            clique = [best_u, best_v]
            # Candidates must be common neighbors of all clique members
            # Start with common neighbors of u and v
            candidates = adj_mask[best_u] & adj_mask[best_v]
            
            while candidates:
                # Pick candidate w that has the most connections to other candidates
                best_w = -1
                max_conn = -1
                c = candidates
                while c:
                    lsb = c & -c
                    w = lsb.bit_length() - 1
                    conn_mask = adj_mask[w] & candidates
                    conn_count = conn_mask.bit_count()
                    if conn_count > max_conn:
                        max_conn = conn_count
                        best_w = w
                    c &= c - 1
                
                if best_w == -1:
                    break
                    
                clique.append(best_w)
                # Update candidates: must be neighbor of best_w
                candidates &= adj_mask[best_w]
                # Remove best_w from candidates
                candidates &= ~(1 << best_w)
                
            # Add clique to result
            cliques.append(sorted(clique))
            
            # Remove edges covered by this clique from uncovered and adj_mask
            for i in range(len(clique)):
                a = clique[i]
                for j in range(i + 1, len(clique)):
                    b = clique[j]
                    e = (min(a, b), max(a, b))
                    if e in uncovered:
                        uncovered.discard(e)
                        # Update adj_masks
                        adj_mask[a] &= ~(1 << b)
                        adj_mask[b] &= ~(1 << a)
                        
        return cliques

    result = []
    for (n, p, seed) in ((32, 0.6, 601), (64, 0.55, 602)):
        edges = random_graph(n, p, seed)
        cliques = partition_cliques(edges)
        result.append(cliques)
    return result
# EVOLVE-BLOCK-END
