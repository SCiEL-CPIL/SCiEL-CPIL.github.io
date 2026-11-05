import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    n = 40
    m_target = 401
    
    def compute_max_book(adj):
        max_book = 0
        for i in range(n):
            mask_i = adj[i]
            for j in range(i + 1, n):
                if (mask_i >> j) & 1:
                    common = (adj[i] & adj[j]).bit_count()
                    if common > max_book:
                        max_book = common
        return max_book

    def get_edge_list(adj):
        edges = []
        for i in range(n):
            mask = adj[i]
            mask &= ~((1 << (i+1)) - 1) 
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                edges.append([i, j])
                mask ^= lsb
        return edges

    def edge_count(adj):
        count = 0
        for i in range(n):
            mask = adj[i]
            mask &= ~((1 << (i+1)) - 1)
            count += mask.bit_count()
        return count

    # Initialization: K_20,20 + Two K_10s
    # Vertices 0-19 are Side A, 20-39 are Side B.
    # Split Side A into A1 (0-9) and A2 (10-19).
    # Split Side B into B1 (20-29) and B2 (30-39).
    
    # Start with K_20,20 (400 edges)
    adj = [0] * n
    for i in range(20):
        for j in range(20, 40):
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            
    # Add K_10 on A1 (0-9): 45 edges
    for i in range(10):
        for j in range(i+1, 10):
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            
    # Add K_10 on B1 (20-29): 45 edges
    for i in range(20, 30):
        for j in range(i+1, 30):
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            
    # Current edge count: 400 + 45 + 45 = 490.
    # We need 401 edges.
    # This initialization has 89 extra edges.
    # We will remove 89 edges to reach 401.
    # Which edges to remove? The ones that contribute most to high books?
    # In K_10, every edge has 8 common neighbors.
    # In K_20,20, every edge has 0 common neighbors (bipartite).
    # Wait, if we have K_20,20 + K_10 + K_10, the books on the clique edges are huge (8).
    # The books on bipartite edges are 0? No, if u in A1, v in B1, and we have cliques on A1 and B1...
    # Common neighbors of (u,v) where u in A1, v in B1:
    # Neighbors of u: all in B (20) + others in A1 (9).
    # Neighbors of v: all in A (20) + others in B1 (9).
    # Intersection: A1 \ {u} is in N(u) but not N(v) (v is in B, N(v) in A is all A, so A1 \ {u} IS in N(v)).
    # Wait, N(v) for v in B1 includes all A (0-19). So A1 \ {u} is in N(v).
    # Is A1 \ {u} in N(u)? Yes, because of K_10.
    # So common neighbors include A1 \ {u} (9 vertices) and B1 \ {v} (9 vertices).
    # Total common neighbors for (u,v) with u in A1, v in B1 is 18.
    # This is very high.
    
    # Strategy: Remove edges to reduce these high books.
    # We need to remove 89 edges.
    # The edges with the highest "pressure" (common neighbors) are the bipartite edges between A1 and B1, and A2 and B2?
    # Actually, let's just use a greedy removal or random removal followed by optimization.
    # Better: Start with K_20,20 (400 edges). Add 1 edge.
    # The previous strategy did this. Why was it 1.8182?
    # Maybe the circulant base was better?
    # Let's stick to the directive: "Model the graph as a K_20,20 with a fixed, non-star internal structure... then use bitmask-accelerated local search to swap edges between the bipartite core and the internal cliques to balance common-neighbor counts."
    
    # The directive suggests starting with the structure described.
    # Let's construct the structure: K_20,20 + K_10(A1) + K_10(B1).
    # Then we must reduce to 401 edges.
    # We will remove edges from the cliques first, as they create high books internally and with their counterparts.
    # Or, we can swap: Remove an edge from the clique, add a random bipartite edge? No, we need to reduce count.
    
    # Let's try a different initialization that is closer to 401 edges but structured.
    # Take K_20,20. Add 1 edge (u,v) within A.
    # This gives 401 edges.
    # The book of (u,v) is 0 (if no other edges in A).
    # The books of edges (u, w) for w in B: common neighbors are 0 (since w in B, N(w) subset A, N(u) subset B + {v}). Intersection is empty?
    # N(u) = B (20) + {v}.
    # N(w) = A (20).
    # Intersection: A \cap (B + {v}) = {v} if v in A. Yes, v is in A.
    # So (u, w) has 1 common neighbor (v).
    # (v, w) has 1 common neighbor (u).
    # Other bipartite edges have 0.
    # So max book is 1.
    # Wait, is that right?
    # If I have K_20,20 + (0,1).
    # Edge (0,1): N(0) = {1} U B. N(1) = {0} U B. Common = B (20 vertices).
    # So the book of (0,1) is 20.
    # This is terrible.
    
    # So adding an edge within a partition of a complete bipartite graph creates a huge book.
    # To minimize the book, we should avoid creating triangles.
    # A graph with no triangles has max book 0.
    # Can we have 401 edges with no triangles?
    # Mantel's theorem: max edges in triangle-free graph on n vertices is n^2/4 = 400.
    # So we MUST have at least one triangle.
    # If we have 401 edges, we have at least 1 triangle.
    # The edges in that triangle will have book size at least 1.
    # Can we keep the max book at 1?
    # If we have exactly 1 triangle, the 3 edges in the triangle have 1 common neighbor each.
    # Other edges might have 0 or more.
    # We need to ensure no edge is in >1 triangle.
    # i.e., the graph should be "linear" in terms of triangles? Or just that no edge is in 2 triangles.
    # If we take K_20,20 and add 1 edge (u,v) in A, we get 1 triangle (u,v,w) for each w in B?
    # No, (u,v,w) is a triangle if u-v, v-w, w-u exist.
    # u-v exists (added).
    # v-w exists (bipartite).
    # w-u exists (bipartite).
    # So for EVERY w in B, (u,v,w) is a triangle.
    # So the edge (u,v) is in 20 triangles. Book size 20.
    # This is why the "K_20,20 + 1 edge" strategy fails if the edge is within a partition.
    
    # How to add 1 edge to K_20,20 without creating 20 triangles?
    # You can't, if you add an edge within a partition.
    # What if we don't start with K_20,20?
    # What if we start with a graph that is ALMOST K_20,20 but modified to allow 1 edge with low book?
    
    # Alternative: Start with a random 401-edge graph and optimize.
    # The current program does this with circulant base.
    # The directive says: "Model the graph as a K_20,20 with a fixed, non-star internal structure... partitioning each side into two groups of 10 and adding all edges within each group (creating two disjoint K_10s), then use bitmask-accelerated local search to swap edges between the bipartite core and the internal cliques to balance common-neighbor counts."
    
    # This specific initialization (K_20,20 + K_10 + K_10) has 490 edges.
    # We need to remove 89 edges.
    # If we remove edges from the cliques, we reduce the book sizes of the bipartite edges connecting the clique vertices.
    # Let's try this specific path.
    
    best_adj = None
    best_score = float('inf')
    
    for seed in range(20):
        if time.time() > deadline - 5:
            break
            
        random.seed(seed + 200)
        
        # Construct Base: K_20,20 + K_10(A1) + K_10(B1)
        adj = [0] * n
        # K_20,20
        for i in range(20):
            for j in range(20, 40):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
        # K_10 on 0-9
        for i in range(10):
            for j in range(i+1, 10):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
        # K_10 on 20-29
        for i in range(20, 30):
            for j in range(i+1, 30):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
        # Current edges: 490. Target: 401. Remove 89.
        # We want to remove edges that reduce the max book.
        # The edges with highest book are likely the bipartite edges between A1 and B1 (book 18) 
        # and the clique edges (book 8).
        # Removing a clique edge (u,v) in A1 reduces the book of (u,v) (if it was 8, it drops)
        # and reduces the book of bipartite edges (u, w) for w in B?
        # N(u) loses v. So common neighbors of (u, w) might decrease if v was a common neighbor.
        # v is in A1. w is in B.
        # Is v in N(w)? Yes, K_20,20.
        # Is v in N(u)? Yes, K_10.
        # So v is a common neighbor of (u, w).
        # So removing (u,v) reduces the book of ALL (u, w) for w in B by 1.
        # This is a good trade-off?
        
        # Let's greedily remove edges with highest "impact" or just random removal + local search.
        # Given time constraints, let's do random removal of 89 edges, then local search.
        
        # List of all edges
        all_edges = []
        for i in range(n):
            mask = adj[i]
            mask &= ~((1 << (i+1)) - 1)
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                all_edges.append((i, j))
                mask ^= lsb
                
        # Remove 89 random edges
        edges_to_remove = random.sample(all_edges, 89)
        for u, v in edges_to_remove:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            
        current_score = compute_max_book(adj)
        if current_score < best_score:
            best_score = current_score
            best_adj = adj.copy()
            
        # Local Search: Swap an edge (u,v) with a non-edge (x,y)
        # Focus on swapping edges that are in high-book triangles with non-edges that don't create high books.
        # Simple random swap SA as before, but starting from this structure.
        
        T = 1.0
        restart_start = time.time()
        
        for step in range(10000):
            if time.time() > deadline - 5:
                break
                
            # Pick random edge
            while True:
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u != v and ((adj[u] >> v) & 1):
                    break
                    
            # Pick random non-edge
            while True:
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                if x != y and not ((adj[x] >> y) & 1):
                    break
                    
            if (u == x and v == y) or (u == y and v == x):
                continue
                
            # Swap
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            adj[x] |= (1 << y)
            adj[y] |= (1 << x)
            
            new_score = compute_max_book(adj)
            
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_adj = adj.copy()
            else:
                if random.random() < np.exp(-1.0 / T):
                    current_score = new_score
                else:
                    # Revert
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    adj[x] &= ~(1 << y)
                    adj[y] &= ~(1 << x)
                    
            T = T_start * (0.5 ** ((time.time() - restart_start) / 5.0))
            if current_score == 1:
                break

    if best_adj is None:
        # Fallback
        adj = [0] * n
        for i in range(20):
            for j in range(20, 40):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        adj[0] |= (1 << 1)
        adj[1] |= (1 << 0)
        best_adj = adj

    return get_edge_list(best_adj)
