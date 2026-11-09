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
        # Use bitmasks for adjacency for fast intersection
        # adj_mask[v] is an integer where bit u is set if (u,v) is an edge
        adj_mask = [0] * n
        
        # Initialize masks from edges
        for u, v in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        
        uncovered = set(edges)
        cliques = []
        
        # Precompute bit counts for efficiency if needed, or just use bit_count()
        
        while uncovered:
            # 1. Identify candidate starting vertices
            # We want vertices with high "clique density".
            # Let's look at the current induced subgraph of uncovered edges.
            # We need to know which vertices are active (have at least one uncovered edge).
            
            active_vertices = set()
            for u, v in uncovered:
                active_vertices.add(u)
                active_vertices.add(v)
                
            # Calculate score for each active vertex
            # Score: (number of edges among neighbors in uncovered graph) / degree_in_uncovered
            # To compute this efficiently, we need the adjacency within the uncovered subgraph.
            
            # Let's build a temporary adjacency list for the uncovered subgraph
            # This is O(E_uncovered)
            temp_adj = {v: set() for v in active_vertices}
            for u, v in uncovered:
                temp_adj[u].add(v)
                temp_adj[v].add(u)
                
            best_clique = []
            best_clique_size = 0
            
            # Sort candidates by potential (degree) to check promising ones first
            # We can limit the number of candidates to check to save time, e.g., top 20
            candidates = sorted(active_vertices, key=lambda x: len(temp_adj[x]), reverse=True)
            
            # Check top K candidates
            K = min(20, len(candidates))
            
            for start in candidates[:K]:
                # Greedily build a maximal clique starting from 'start'
                # Current clique
                clique = [start]
                # Candidates must be neighbors of all current clique members in the UNCOVERED graph
                # Start with neighbors of start
                current_candidates = set(temp_adj[start])
                
                # Iterate to add best candidates
                while current_candidates:
                    # Find candidate in current_candidates that is connected to all in clique
                    # AND has highest degree within the remaining valid candidates
                    valid = []
                    for c in current_candidates:
                        # Check if c is connected to all in clique
                        is_connected = True
                        for m in clique:
                            if c not in temp_adj.get(m, set()):
                                is_connected = False
                                break
                        if is_connected:
                            valid.append(c)
                            
                    if not valid:
                        break
                    
                    # Pick the valid candidate with highest degree in the subgraph induced by valid + current_candidates?
                    # Or just highest degree in temp_adj?
                    # To maximize clique size, pick the one with most connections to other valid candidates.
                    best_c = None
                    max_conn = -1
                    for c in valid:
                        # Count connections to other valid candidates
                        conn = 0
                        for d in valid:
                            if c != d and d in temp_adj.get(c, set()):
                                conn += 1
                        if conn > max_conn:
                            max_conn = conn
                            best_c = c
                            
                    if best_c is None:
                        break
                        
                    clique.append(best_c)
                    
                    # Update current_candidates:
                    # Must be in valid, and connected to best_c
                    new_candidates = set()
                    for c in valid:
                        if c != best_c and best_c in temp_adj.get(c, set()):
                            new_candidates.add(c)
                    current_candidates = new_candidates
                    
                if len(clique) > best_clique_size:
                    best_clique_size = len(clique)
                    best_clique = clique
                    
            # If no clique found (should not happen if uncovered is not empty), fallback to single edge
            if not best_clique:
                e = next(iter(uncovered))
                best_clique = [min(e), max(e)]
            
            # Add the best clique found
            if len(best_clique) >= 2:
                cliques.append(sorted(best_clique))
                # Remove edges covered by this clique
                for i in range(len(best_clique)):
                    a = best_clique[i]
                    for j in range(i + 1, len(best_clique)):
                        b = best_clique[j]
                        e = (min(a, b), max(a, b))
                        uncovered.discard(e)
            else:
                # Should not reach here if best_clique logic is correct, but safety check
                e = next(iter(uncovered))
                cliques.append([min(e), max(e)])
                uncovered.discard(e)
                
        return cliques

    result = []
    for (n, p, seed) in ((32, 0.6, 601), (64, 0.55, 602)):
        edges = random_graph(n, p, seed)
        cliques = partition_cliques(edges)
        result.append(cliques)
    return result
# EVOLVE-BLOCK-END
