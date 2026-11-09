"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    deadline = time.time() + 130
    
    # Start with the known best: 8 disjoint K4's
    best_n = 32
    best_edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                best_edges.append([base + i, base + j])
    
    rng = random.Random(42)
    
    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def has_c5_fast(adj, n):
        """Fast C5 check using adjacency sets."""
        # For each ordered pair (i, j) with i < j, non-adjacent:
        # A 5-cycle through i,j exists if there are a in N(i), c in N(j), 
        # a != c, a not adj c, and N(a) ∩ N(c) non-empty (with the middle vertex distinct)
        # More carefully: cycle i-a-b-c-j-i requires:
        #   a in N(i), c in N(j), b in N(a) ∩ N(c), b != i, b != j, b != a, b != c
        #   and a != c, a not in N(c) (to avoid 4-cycle shortcut, though not strictly needed)
        for i in range(n):
            for j in range(i + 1, n):
                if j in adj[i]:
                    continue
                ni = adj[i]
                nj = adj[j]
                for a in ni:
                    if a == j:
                        continue
                    for c in nj:
                        if c == i or c == a:
                            continue
                        if c in adj[a]:
                            continue
                        # Need b in N(a) ∩ N(c), b not in {i, j, a, c}
                        common = adj[a] & adj[c]
                        if len(common) >= 1:
                            for b in common:
                                if b != i and b != j and b != a and b != c:
                                    return True
        return False
    
    def has_indep_set_k(adj, n, k):
        """Backtracking: is there an independent set of size k?"""
        # Use bitmasks for speed
        # candidates bitmask
        def bt(cands_mask, depth):
            if depth == k:
                return True
            # Lower bound: need at least k-depth vertices in cands
            if cands_mask.bit_count() < k - depth:
                return False
            # Pick vertex with fewest neighbors in candidates (MRV heuristic)
            # Find a vertex in cands
            v = (cands_mask & -cands_mask).bit_length() - 1
            # Option 1: include v
            new_cands = cands_mask & ~((1 << v) | adj_mask[v])
            if bt(new_cands, depth + 1):
                return True
            # Option 2: exclude v
            if bt(cands_mask & ~(1 << v), depth):
                return True
            return False
        
        adj_mask = [0] * n
        for u in range(n):
            for v in adj[u]:
                adj_mask[u] |= (1 << v)
        
        return bt((1 << n) - 1, 0)
    
    def is_valid(n, edges):
        """Check C5-free and alpha < 9."""
        adj = build_adj(n, edges)
        if has_c5_fast(adj, n):
            return False
        if has_indep_set_k(adj, n, 9):
            return False
        return True
    
    # Local search: start from 8xK4 on 32 vertices, try to extend to 33
    # Strategy: take the 32-vertex base, add vertex 32, try different edge sets
    # Also try random restarts with local search
    
    def local_search_from_base(n_target, base_edges, max_iters=200):
        """Local search: start from base_edges (on n_target vertices), flip edges."""
        edges = list(base_edges)
        edge_set = set((min(u,v), max(u,v)) for u, v in edges)
        
        # Add vertex n_target-1 with some initial edges
        # Try to find a valid configuration
        current_valid = False
        if n_target == 33:
            # Start with vertex 32 connected to some vertices
            # Try all subsets of size 0-4 from the 32 vertices
            best_found = None
            for subset_size in range(0, 5):
                if time.time() > deadline - 10:
                    break
                # Try random subsets
                for trial in range(50):
                    if time.time() > deadline - 10:
                        break
                    subset = set(rng.sample(range(32), subset_size))
                    edges_try = list(base_edges)
                    for v in subset:
                        edges_try.append([32, v])
                    if is_valid(33, edges_try):
                        return edges_try
        return None
    
    # Try structured approaches for n=33
    # Idea: take 7 K4's (28 vertices) + a C5-free graph on 5 vertices with alpha <= 1
    # K5 has C5, so that doesn't work.
    # 7 K4's + K4 = 8 K4's = 32 vertices, alpha=8
    # To get 33: 7 K4's (28, alpha=7) + component on 5 vertices with alpha <= 1
    # Component on 5 vertices with alpha=1 means K5, which has C5. Not possible.
    # So we need: components with total alpha <= 8 and total n = 33
    # e.g., 7 K4's (alpha=7) + 5-vertex component with alpha=1 -> impossible (K5 has C5)
    # e.g., 6 K4's (alpha=6) + 9-vertex component with alpha=2, C5-free
    #   A 9-vertex C5-free graph with alpha=2: e.g., complete bipartite K_{4,5} has alpha=5, no.
    #   Actually alpha(K_{a,b}) = max(a,b). So we need max(a,b) <= 2, meaning a,b <= 2, n<=4.
    #   What about other C5-free graphs with alpha=2 on 9 vertices?
    #   A graph with alpha=2 means complement has no K2, i.e., complement is edgeless, so graph is complete.
    #   Wait no: alpha(G) = 2 means no two non-adjacent vertices, so G is complete. K9 has C5.
    #   So no 9-vertex C5-free graph has alpha=2.
    #   
    # Let me reconsider. We need sum of alphas of components <= 8.
    # For a C5-free component on m vertices, what's the minimum possible alpha?
    # - m=4: K4, alpha=1
    # - m=5: minimum alpha for C5-free? K5 has C5. 
    #   What about K5 minus an edge? Still has C5. 
    #   Actually any 5-vertex graph with alpha=1 is K5 which has C5.
    #   So min alpha for 5-vertex C5-free is 2.
    #   Example: K5 minus a matching of size 1 (remove one edge). Does it have C5?
    #   Vertices 0,1,2,3,4. Remove edge (0,1). Cycle 0-2-1-3-4-0? Edge 0-2 yes, 2-1 yes, 1-3 yes, 3-4 yes, 4-0 yes. Yes, has C5.
    #   What about the complement of C5? That's also C5. 
    #   What about K_{2,3}? alpha=3.
    #   What about a 5-vertex graph that's C5-free with alpha=2?
    #   If alpha=2, the complement has no edges (since alpha(G)=2 means G's complement has no K2, wait...).
    #   alpha(G) = omega(complement(G)). So alpha(G)=2 means complement(G) has no K3? No.
    #   alpha(G) is the size of largest independent set in G = size of largest clique in complement(G).
    #   So alpha(G)=2 means complement(G) has no triangle, i.e., complement is triangle-free.
    #   And G is C5-free means complement(G) is also C5-free (complement of C5 is C5).
    #   So we need a 5-vertex graph H (the complement) that is triangle-free and C5-free.
    #   Triangle-free 5-vertex graphs: max edges is floor(25/4)=6 (Mantel).
    #   C5-free: H must not contain C5.
    #   H has 5 vertices, triangle-free, no C5. 
    #   If H has 5 vertices and no C5, and is triangle-free... 
    #   A 5-vertex triangle-free graph with no C5: 
    #   - If it has a C4, say 0-1-2-3-0, then vertex 4 must connect to at most... 
    #   - Star K_{1,4}: triangle-free, no C5 (only 5 vertices, C5 needs 5-cycle). K_{1,4} has no C5. 
    #     So H = K_{1,4} works. Then G = complement of K_{1,4} = K4 + isolated vertex? 
    #     No: complement of star K_{1,4} on 5 vertices: center is isolated, leaves form K4. 
    #     So G = K4 ∪ K1. alpha(G) = alpha(K4) + alpha(K1) = 1 + 1 = 2. 
    #     And G is C5-free (disconnected, components are K4 and K1). 
    #     So a 5-vertex C5-free graph with alpha=2 exists: K4 ∪ K1.
    #     
    # So for 33 vertices: 7 K4's (28 vertices, alpha=7) + (K4 ∪ K1) (5 vertices, alpha=2) = 33 vertices, alpha=9. 
    # But we need alpha < 9, i.e., alpha <= 8. This gives alpha=9, which is too high.
    # 
    # What about 6 K4's (24 vertices, alpha=6) + 9-vertex C5-free component with alpha <= 2?
    # As argued, no 9-vertex C5-free graph has alpha=2 (would need to be complete, which has C5).
    # 
    # What about 6 K4's (24, alpha=6) + 9-vertex component with alpha=2? Not possible.
    # 
    # Let's think differently. Maybe a non-disjoint-union approach.
    # What if we take a C5-free graph that's not a disjoint union of cliques?
    # 
    # Key insight: The bound R(C5, K9) > 32 is known. Whether R(C5, K9) > 33 is the question.
    # Let's just do a thorough random + local search for n=33.
    
    # Random search for n=33
    attempts = 0
    while time.time() < deadline - 15 and attempts < 3000:
        attempts += 1
        
        n_try = 33
        # Generate random sparse graph
        p = rng.uniform(0.15, 0.45)
        edges_try = []
        for i in range(n_try):
            for j in range(i + 1, n_try):
                if rng.random() < p:
                    edges_try.append([i, j])
        
        adj = build_adj(n_try, edges_try)
        if has_c5_fast(adj, n_try):
            continue
        if has_indep_set_k(adj, n_try, 9):
            continue
        
        best_n = n_try
        best_edges = edges_try
        break
    
    # If not found via pure random, try local search from structured bases
    if best_n == 32 and time.time() < deadline - 15:
        # Try: 7 K4's + 5-vertex component, optimize the 5-vertex part
        # 7 K4's = 28 vertices, alpha=7. Need 5 more vertices with alpha <= 1.
        # Impossible as shown. So try different decomposition.
        # 
        # Try: 8 K4's (32 vertices) + 1 vertex. The 33rd vertex must connect to enough
        # vertices to keep alpha < 9. In 8 K4's, alpha=8 (one from each K4).
        # Adding vertex 32: to keep alpha < 9, vertex 32 must be adjacent to at least one
        # vertex in each K4 (so that any independent set of size 9 would need to include
        # vertex 32 and 8 others, but vertex 32 blocks at least one choice per K4).
        # Actually: if vertex 32 is adjacent to vertex v_b in block b, then an independent set
        # containing vertex 32 cannot contain v_b. But it could still contain other vertices
        # from block b. Since each K4 has alpha=1, an independent set from block b has exactly
        # 1 vertex. If we exclude v_b, we can still pick one of the other 3.
        # So alpha(8 K4's + vertex 32) = max(8, 1 + 8) = 9 if vertex 32 is isolated, or
        # if vertex 32 connects to some vertices, the alpha is still 8 or 9.
        # Specifically: any independent set either includes 32 or not.
        # If not: alpha = 8 (one from each K4).
        # If yes: 1 + (independent set from remaining 32 vertices avoiding N(32)).
        # If N(32) has at least one vertex in each K4, then from each K4 we can still pick
        # 1 vertex (avoiding the neighbor of 32), so alpha = 1 + 8 = 9.
        # If N(32) has no vertex in some K4, then from that K4 we pick 1, and from others
        # we pick 1 (avoiding neighbors), so alpha = 1 + 8 = 9 still.
        # Wait, if 32 is adjacent to ALL vertices in some K4, then from that K4 we pick 0.
        # So alpha = 1 + (sum over blocks of min(1, 4 - |N(32) ∩ block|)).
        # To get alpha < 9, we need 1 + sum(...) < 9, i.e., sum(...) < 8.
        # sum(...) <= 8 always (max 1 per block). So we need sum(...) <= 7.
        # This means for at least one block, |N(32) ∩ block| >= 1, AND for the sum to be <= 7,
        # we need at least one block where |N(32) ∩ block| >= 2? No:
        # sum(...) = sum over blocks of min(1, 4 - |N(32) ∩ block|)
        # If |N(32) ∩ block| = 0: contributes 1
        # If |N(32) ∩ block| = 1: contributes 1 (can pick 1 of remaining 3)
        # If |N(32) ∩ block| = 2: contributes 1 (can pick 1 of remaining 2)
        # If |N(32) ∩ block| = 3: contributes 1 (can pick 1 of remaining 1)
        # If |N(32) ∩ block| = 4: contributes 0
        # So sum(...) = 8 - (number of blocks where |N(32) ∩ block| = 4)
        # For alpha < 9: 1 + 8 - (#blocks fully covered) < 9 => #blocks fully covered > 0.
        # So vertex 32 must be adjacent to ALL 4 vertices of at least one K4.
        # But then vertex 32 + that K4 forms K5, which contains C5!
        # So this approach fails.
        #
        # What if we modify the base? Instead of 8 K4's, use a different structure.
        # Let's try: take the 32-vertex 8xK4 graph and try to add vertex 32 with various
        # connections, while also modifying some edges in the base to avoid C5.
        
        # Try random modifications of 8xK4 + vertex 32
        base_edges_32 = []
        for b in range(8):
            base = 4 * b
            for i in range(4):
                for j in range(i + 1, 4):
                    base_edges_32.append([base + i, base + j])
        
        for trial in range(500):
            if time.time() > deadline - 15:
                break
            
            # Start with 8xK4, add vertex 32 with random connections
            edges_try = list(base_edges_32)
            # Connect vertex 32 to random subset of 32 vertices
            num_conn = rng.randint(4, 12)
            conn_targets = rng.sample(range(32), num_conn)
            for v in conn_targets:
                edges_try.append([32, v])
            
            adj = build_adj(33, edges_try)
            if has_c5_fast(adj, 33):
                continue
            if has_indep_set_k(adj, 33, 9):
                continue
            
            best_n = 33
            best_edges = edges_try
            break
        
        # If still not found, try more aggressive random search
        if best_n == 32 and time.time() < deadline - 10:
            for trial in range(2000):
                if time.time() > deadline - 10:
                    break
                n_try = 33
                p = rng.uniform(0.1, 0.5)
                edges_try = []
                for i in range(n_try):
                    for j in range(i + 1, n_try):
                        if rng.random() < p:
                            edges_try.append([i, j])
                
                adj = build_adj(n_try, edges_try)
                if has_c5_fast(adj, n_try):
                    continue
                if has_indep_set_k(adj, n_try, 9):
                    continue
                
                best_n = n_try
                best_edges = edges_try
                break
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
