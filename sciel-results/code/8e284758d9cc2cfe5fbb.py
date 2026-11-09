"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Construct a graph with 40 vertices and 401 edges that minimizes the largest book size.
    
    Strategy: Start with K_{20,20}, remove k bipartite edges, add k+1 intra-part edges.
    Use greedy selection to balance book sizes.
    """
    n = 40
    part_a = list(range(20))
    part_b = list(range(20, 40))
    
    # Start with all bipartite edges
    edges = set()
    for a in part_a:
        for b in part_b:
            edges.add((a, b))
    
    # We'll remove some bipartite edges and add intra-part edges
    # Try different numbers of intra-edges to find the best balance
    
    def compute_max_book(edge_set):
        """Compute the maximum book size in the graph."""
        adj = {i: set() for i in range(n)}
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        max_book = 0
        for u, v in edge_set:
            common = len(adj[u] & adj[v])
            if common > max_book:
                max_book = common
        return max_book
    
    # Try different strategies: add m intra-A edges and n intra-B edges
    # Remove k = m + n - 1 bipartite edges
    
    best_max_book = 20
    best_edges = None
    
    # Try various combinations
    for num_intra_a in range(1, 12):
        for num_intra_b in range(0, 12 - num_intra_a):
            k = num_intra_a + num_intra_b - 1
            if k < 0:
                continue
            
            # Create intra-A edges as a matching (each vertex used at most once)
            intra_a_edges = []
            used_a = set()
            for i in range(num_intra_a):
                u = 2 * i
                v = 2 * i + 1
                if u < 20 and v < 20:
                    intra_a_edges.append((u, v))
                    used_a.add(u)
                    used_a.add(v)
            
            # Create intra-B edges as a matching
            intra_b_edges = []
            used_b = set()
            for i in range(num_intra_b):
                u = 20 + 2 * i
                v = 20 + 2 * i + 1
                if u < 40 and v < 40:
                    intra_b_edges.append((u, v))
                    used_b.add(u)
                    used_b.add(v)
            
            # Now remove k bipartite edges to balance book sizes
            # For each intra-A edge (u,v), its book size = common neighbors in B
            # We want to remove edges to reduce this
            
            # Build the edge set
            candidate_edges = set()
            for a in part_a:
                for b in part_b:
                    candidate_edges.add((a, b))
            
            # Remove some bipartite edges
            # Greedy: for each intra-edge, remove bipartite edges incident to its endpoints
            # to reduce common neighbors
            
            # Let's try removing edges in a structured way
            # For intra-A edge (u,v), remove edges (u,b) and (v,b) for some b in B
            # This reduces the common neighbors
            
            removed = set()
            # We need to remove exactly k edges
            # Choose which edges to remove to minimize max book size
            
            # Simple heuristic: remove edges incident to vertices in intra-edges
            # Distribute removals evenly
            
            # For each intra-A edge, we can remove up to 2 edges per endpoint
            # Let's remove edges to make book sizes roughly equal
            
            # Build candidate edge set
            candidate_edges = set()
            for a in part_a:
                for b in part_b:
                    candidate_edges.add((a, b))
            
            # Remove k bipartite edges
            # Priority: remove edges that are common neighbors of multiple intra-edges
            # For simplicity, remove edges in a round-robin fashion from endpoints of intra-edges
            
            endpoints_a = set()
            for u, v in intra_a_edges:
                endpoints_a.add(u)
                endpoints_a.add(v)
            
            endpoints_b = set()
            for u, v in intra_b_edges:
                endpoints_b.add(u)
                endpoints_b.add(v)
            
            # Remove edges: for each endpoint in A, remove some edges to B
            # For each endpoint in B, remove some edges to A
            
            # Let's try: for each intra-A edge (u,v), remove (u, b) and (v, b) for b in some subset
            # This makes b not a common neighbor
            
            # Greedy removal:
            # Start with all bipartite edges
            # For each intra-A edge, we want to reduce its book size
            # Remove edges (u, b) for b in B - this removes b as neighbor of u
            # But this also affects other edges
            
            # Let's just try a simple pattern: remove edges in a cyclic pattern
            # For simplicity, remove k edges from the bipartite graph in a structured way
            
            # Remove edges: (a, b) where a in endpoints_a or b in endpoints_b
            # Pick k such edges
            
            removable = []
            for a in part_a:
                for b in part_b:
                    if a in endpoints_a or b in endpoints_b:
                        removable.append((a, b))
            
            # If not enough, add more
            if len(removable) < k:
                for a in part_a:
                    for b in part_b:
                        if (a, b) not in set(removable):
                            removable.append((a, b))
                            if len(removable) >= k:
                                break
                    if len(removable) >= k:
                        break
            
            # Remove the first k edges
            for i in range(min(k, len(removable))):
                candidate_edges.discard(removable[i])
            
            # Add intra edges
            for u, v in intra_a_edges:
                candidate_edges.add((u, v))
            for u, v in intra_b_edges:
                candidate_edges.add((u, v))
            
            # Check edge count
            if len(candidate_edges) != 401:
                continue
            
            # Compute max book size
            mb = compute_max_book(candidate_edges)
            
            if mb < best_max_book:
                best_max_book = mb
                best_edges = set(candidate_edges)
            
            if best_max_book <= 7:
                break
        if best_max_book <= 7:
            break
    
    if best_edges is None:
        # Fallback: original construction
        best_edges = set()
        for a in range(20):
            for b in range(20, 40):
                best_edges.add((a, b))
        best_edges.add((0, 1))
    
    result = [[u, v] for u, v in best_edges]
    return result

# EVOLVE-BLOCK-END
