import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    n = 16
    # Adjacency matrix as list of bitmasks
    adj = [0] * n
    
    def get_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    def has_q3_with_edge(u, v, adj):
        """
        Check if the graph (with edge u-v present) contains a Q3 that uses edge (u,v).
        In Q3, u and v are adjacent. WLOG u=0, v=1 in the Q3 template.
        Q3 structure:
          0: neighbors {1,2,3}
          1: neighbors {0,4,5}
          2: neighbors {0,4,6}
          3: neighbors {0,5,6}
          4: neighbors {1,2,7}
          5: neighbors {1,3,7}
          6: neighbors {2,3,7}
          7: neighbors {4,5,6}
        """
        mask_uv = (1 << u) | (1 << v)
        neu = adj[u] & ~mask_uv
        nev = adj[v] & ~mask_uv
        
        neu_list = [i for i in range(n) if (neu >> i) & 1]
        nev_list = [i for i in range(n) if (nev >> i) & 1]
        
        if len(neu_list) < 2 or len(nev_list) < 2:
            return False
        
        # Precompute common neighbors for speed
        # For each pair (a,b) from neu_list, find vertices adjacent to both
        # For each pair (c,d) from nev_list, find vertices adjacent to both
        
        for a, b in itertools.combinations(neu_list, 2):
            # a,b map to 2,3 in Q3
            # Find common neighbors of a and b (candidates for 6)
            common_ab = adj[a] & adj[b]
            # Exclude u, v, a, b themselves
            mask_ab = (1 << a) | (1 << b)
            common_ab &= ~mask_ab
            common_ab &= ~(1 << u) & ~(1 << v)
            
            for c, d in itertools.combinations(nev_list, 2):
                if c == a or c == b or d == a or d == b:
                    continue
                # c,d map to 4,5 in Q3
                # Check if a-c and b-d, or a-d and b-c
                if (adj[a] & (1 << c)) and (adj[b] & (1 << d)):
                    # c=4, d=5
                    # e must be adjacent to a(2), b(3) -> e in common_ab
                    # f must be adjacent to c(4), d(5)
                    common_cd = adj[c] & adj[d]
                    mask_cd = (1 << c) | (1 << d)
                    common_cd &= ~mask_cd
                    common_cd &= ~(1 << u) & ~(1 << v)
                    common_cd &= ~(1 << a) & ~(1 << b)
                    
                    # e from common_ab, f from common_cd, e!=f, e-f edge
                    e_mask = common_ab & ~(1 << c) & ~(1 << d)
                    f_mask = common_cd & ~(1 << a) & ~(1 << b)
                    
                    # For each e in e_mask, check if any f in f_mask is adjacent to e
                    e_bits = e_mask
                    while e_bits:
                        e = e_bits & (-e_bits)
                        e_idx = e.bit_length() - 1
                        if adj[e_idx] & f_mask:
                            return True
                        e_bits &= e_bits - 1
                elif (adj[a] & (1 << d)) and (adj[b] & (1 << c)):
                    # c=5, d=4
                    common_cd = adj[c] & adj[d]
                    mask_cd = (1 << c) | (1 << d)
                    common_cd &= ~mask_cd
                    common_cd &= ~(1 << u) & ~(1 << v)
                    common_cd &= ~(1 << a) & ~(1 << b)
                    
                    e_mask = common_ab & ~(1 << c) & ~(1 << d)
                    f_mask = common_cd & ~(1 << a) & ~(1 << b)
                    
                    e_bits = e_mask
                    while e_bits:
                        e = e_bits & (-e_bits)
                        e_idx = e.bit_length() - 1
                        if adj[e_idx] & f_mask:
                            return True
                        e_bits &= e_bits - 1
        return False

    # Initialize with greedy
    # Start with empty graph? Or dense?
    # Let's start with the greedy result from the seed to have a good start.
    # But implementing the seed's greedy is redundant if we just run SA from a random dense graph.
    # However, a good start helps.
    
    # Let's try to build a dense graph quickly.
    # Random initialization
    random.seed(42)
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.8:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Remove any Q3s initially
    # Simple way: check all edges, if removing one breaks a Q3, keep it?
    # Actually, just check if current graph has Q3. If yes, remove random edges until valid.
    def has_any_q3(adj):
        # Check all pairs of edges? No, check all 8-subsets is too slow.
        # Use the fact that we can check incrementally?
        # For initialization, we can just try to remove edges.
        # Or better: Start with K16 and remove edges?
        # Let's stick to the "Add/Kick" strategy.
        
        # To ensure we start valid, let's run a quick greedy add.
        # Or just assume we start with a graph that might have Q3s and fix it.
        pass

    # Let's restart with a proper initialization:
    # Start with K16 and remove edges until Q3-free?
    # Or start empty and add?
    # The seed used K3,13 + greedy.
    # Let's use a random dense graph and "fix" it by removing edges involved in Q3s.
    
    # To find if a graph has Q3, we can iterate over all 8-subsets.
    # 12870 subsets. For each, check isomorphism.
    # Isomorphism check for 8 vertices is fast.
    
    def is_q3_subset(vertices, adj):
        # vertices is a tuple of 8 indices
        # Map to 0..7
        # Check all 12 edges of Q3
        # Q3 edges:
        # (0,1), (0,2), (0,3), (1,4), (1,5), (2,4), (2,6), (3,5), (3,6), (4,7), (5,7), (6,7)
        # We need to check if there is a permutation of vertices that satisfies these.
        # Since 8! = 40320, and we have 12870 subsets, total ~500M checks. Too slow in Python.
        
        # Better: Use the structure.
        # Pick a vertex in the subset, say v0.
        # Its neighbors in the subset must be exactly 3.
        # If any vertex has degree != 3 in the induced subgraph, return False.
        
        # Induced subgraph degrees
        degs = [0]*8
        for i in range(8):
            for j in range(8):
                if i == j: continue
                if adj[vertices[i]] & (1 << vertices[j]):
                    degs[i] += 1
        if any(d != 3 for d in degs):
            return False
            
        # Now check bipartiteness? Q3 is bipartite.
        # And check specific structure.
        # Actually, if a 3-regular bipartite graph on 8 vertices is not Q3, it could be K4,4? No, K4,4 has degree 4.
        # The only 3-regular bipartite graph on 8 vertices is Q3?
        # Let's check.
        # 3-regular graphs on 8 vertices:
        # Q3 is one.
        # Are there others?
        # The complement of Q3 is 4-regular.
        # Yes, Q3 is the unique 3-regular bipartite graph on 8 vertices?
        # Wait, the cube Q3 is bipartite.
        # Any 3-regular bipartite graph on 8 vertices must be Q3?
        # Let's assume yes for now, but verify.
        # If it's 3-regular and bipartite, it's a candidate.
        # We should verify it's isomorphic to Q3.
        # Since it's small, we can just check if it contains a 4-cycle? Q3 has many.
        # Actually, the property "3-regular bipartite on 8 vertices" might be sufficient to identify Q3 in this context?
        # Let's double check.
        # There are 2 non-isomorphic 3-regular graphs on 8 vertices?
        # No, Q3 is the only 3-regular bipartite graph on 8 vertices?
        # Actually, there is another one: The "Möbius ladder" M4? No, M4 is 3-regular on 8 vertices but not bipartite (has odd cycles).
        # So if it's bipartite and 3-regular, it is Q3.
        
        # Check bipartiteness
        color = [-1]*8
        color[0] = 0
        queue = [0]
        while queue:
            u = queue.pop(0)
            for v in range(8):
                if adj[vertices[u]] & (1 << vertices[v]):
                    if color[v] == -1:
                        color[v] = color[u] ^ 1
                        queue.append(v)
                    elif color[v] == color[u]:
                        return False
        return True

    def has_any_q3_fast(adj):
        for subset in itertools.combinations(range(n), 8):
            if is_q3_subset(subset, adj):
                return True
        return False

    # Initialization: Start with K16 and remove edges?
    # K16 has many Q3s.
    # Let's start with the greedy result from the seed logic to be safe.
    # But we can just start with a random graph and remove edges until valid.
    
    adj = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.9:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Remove edges to make it Q3-free
    # This is a heuristic initialization.
    # While has_any_q3_fast(adj):
    #    Find a Q3, remove a random edge from it.
    # This is slow.
    
    # Better: Use the greedy add strategy from seed but with bitmasks.
    # Start empty.
    adj = [0] * n
    edges_to_try = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(edges_to_try)
    
    for u, v in edges_to_try:
        if time.time() > deadline: break
        # Check if adding (u,v) creates Q3
        # Tentatively add
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        if has_q3_with_edge(u, v, adj):
            # Revert
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            # Keep
            pass
            
    # Now we have a valid graph.
    # Run Local Search
    best_edges = sum(bin(adj[i]).count('1') for i in range(n)) // 2
    current_edges = best_edges
    best_adj = adj[:]
    
    # Precompute all missing edges for faster access
    def get_missing_edges(adj):
        missing = []
        for i in range(n):
            for j in range(i+1, n):
                if not (adj[i] & (1 << j)):
                    missing.append((i, j))
        return missing
    
    def get_existing_edges(adj):
        existing = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    existing.append((i, j))
        return existing
    
    while time.time() < deadline:
        # Try to add a random missing edge
        added = False
        missing = get_missing_edges(adj)
        random.shuffle(missing)
        
        for u, v in missing:
            if time.time() > deadline: break
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if has_q3_with_edge(u, v, adj):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                current_edges += 1
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj[:]
                added = True
                break
        
        if not added:
            # Local optimum. Kick: remove 1-2 random edges
            existing = get_existing_edges(adj)
            if not existing:
                break
            
            num_remove = random.randint(1, 2)
            for _ in range(min(num_remove, len(existing))):
                u, v = random.choice(existing)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                current_edges -= 1
                existing.remove((u, v))

    # Return best found
    return get_edges(best_adj)
