<<<<<<< SEARCH
def construct():
    deadline = time.time() + 115  # Safety margin
    n = 18
    
    # Seed graph: K6,6,6
    # Parts: 0-5, 6-11, 12-17
    seed_adj = [0] * n
    for i in range(6):
        for j in range(6, 12):
            seed_adj[i] |= (1 << j)
            seed_adj[j] |= (1 << i)
        for j in range(12, 18):
            seed_adj[i] |= (1 << j)
            seed_adj[j] |= (1 << i)
    for i in range(6, 12):
        for j in range(12, 18):
            seed_adj[i] |= (1 << j)
            seed_adj[j] |= (1 << i)
            
    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edges.append([u, v])
        return edges

    best_edges = adj_to_edges(seed_adj)
    best_score = 12 # K6,6,6 has score 12
    
    # Precompute all masks for sizes 12, 11, 10 to speed up evaluation
    # However, generating them on the fly in the check function above is fine 
    # because the check function is called infrequently relative to edge additions?
    # No, we call it for every candidate graph.
    # Let's precompute the masks once.
    
    masks_12 = []
    for combo in combinations(range(n), 12):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        masks_12.append(mask)
        
    masks_11 = []
    for combo in combinations(range(n), 11):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        masks_11.append(mask)
        
    masks_10 = []
    for combo in combinations(range(n), 10):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        masks_10.append(mask)

    def fast_is_triangle_free(mask, adj):
        """Check if subgraph induced by mask is triangle-free."""
        for u in range(n):
            if not (mask & (1 << u)):
                continue
            nu = adj[u] & mask
            v_mask = nu
            while v_mask:
                lsb = v_mask & -v_mask
                v = lsb.bit_length() - 1
                v_mask ^= lsb
                if (adj[u] & adj[v] & mask) != 0:
                    return False
        return True

    def evaluate_score(adj):
        # Check 12
        for mask in masks_12:
            if fast_is_triangle_free(mask, adj):
                return 12
        # Check 11
        for mask in masks_11:
            if fast_is_triangle_free(mask, adj):
                return 11
        # Check 10
        for mask in masks_10:
            if fast_is_triangle_free(mask, adj):
                return 10
        return 9

    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) keeps graph K4-free."""
        common = adj[u] & adj[v]
        c_mask = common
        while c_mask:
            lsb = c_mask & -c_mask
            w = lsb.bit_length() - 1
            c_mask ^= lsb
            if (adj[w] & common) != 0:
                return False
        return True

    def generate_random_max_k4_free(rng):
        adj = [0] * n
        edges_to_try = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(edges_to_try)
        
        for u, v in edges_to_try:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    rng = random.Random(42)
    
    # Try random graphs
    trials = 0
    while time.time() < deadline:
        trials += 1
        adj = generate_random_max_k4_free(rng)
        
        score = evaluate_score(adj)
        
        if score < best_score:
            best_score = score
            best_edges = adj_to_edges(adj)
            if best_score <= 10:
                # Found a very good graph, return immediately
                return best_edges
                
    return best_edges

# EVOLVE-BLOCK-END
=======
def construct():
    deadline = time.time() + 115  # Safety margin
    
    # Precompute masks
    masks_12, masks_12_edge_sets = _precompute_masks()
    
    def adj_to_edges(adj):
        edges = []
        for u in range(N):
            for v in range(u + 1, N):
                if adj[u] & (1 << v):
                    edges.append([u, v])
        return edges
    
    # Start with K6,6,6 (complete tripartite)
    # Parts: 0-5, 6-11, 12-17
    adj = [0] * N
    for i in range(6):
        for j in range(6, 12):
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
        for j in range(12, 18):
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
    for i in range(6, 12):
        for j in range(12, 18):
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
    
    # Verify K4-free
    if not _is_k4_free_adj(adj):
        raise ValueError("Initial graph is not K4-free")
    
    # Compute initial score
    initial_tf_count = _count_tf_12(adj, masks_12, masks_12_edge_sets)
    initial_score = _max_tf_size_exact(adj)
    
    best_adj = adj[:]
    best_score = initial_score
    best_tf_count = initial_tf_count
    best_edges = adj_to_edges(adj)
    
    if best_score <= 11:
        return best_edges
    
    # Local search: try flipping edges to reduce number of TF 12-subsets
    # While maintaining K4-freeness
    
    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) keeps graph K4-free."""
        common = adj[u] & adj[v]
        c_mask = common
        while c_mask:
            lsb = c_mask & -c_mask
            w = lsb.bit_length() - 1
            c_mask ^= lsb
            if (adj[w] & common) != 0:
                return False
        return True
    
    def can_remove_edge(adj, u, v):
        """Removing an edge always keeps K4-free (removing edges can't create K4)."""
        return True
    
    # Simulated annealing / local search
    rng = random.Random(42)
    current_adj = adj[:]
    current_tf_count = initial_tf_count
    current_score = initial_score
    
    # Precompute edge index mapping
    edge_idx = {}
    for ei, (u, v) in enumerate(ALL_EDGES):
        edge_idx[(u, v)] = ei
        edge_idx[(v, u)] = ei
    
    # For efficiency, we track which edges are present
    # and use incremental updates where possible
    
    iterations = 0
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        iterations += 1
        
        # Pick a random edge to flip
        ei = rng.randint(0, NUM_EDGES - 1)
        u, v = ALL_EDGES[ei]
        
        has_edge = bool(current_adj[u] & (1 << v))
        
        if has_edge:
            # Try removing the edge
            # Removing an edge can't create K4, so always valid
            # But it might increase TF count (removing edges makes it easier to be triangle-free)
            # So we only want to remove if it helps... actually removing edges INCREASES
            # the number of triangle-free subsets. So we should AVOID removing edges.
            # Only add edges (or keep) to reduce TF count.
            # Actually, let's think more carefully:
            # Adding an edge can create triangles, reducing TF subsets.
            # Removing an edge removes triangles, potentially increasing TF subsets.
            # So we want to ADD edges, not remove them.
            # But we need to maintain K4-freeness.
            # Let's skip removals for now and focus on additions.
            continue
        
        # Try adding the edge
        if not can_add_edge(current_adj, u, v):
            continue
        
        # Add the edge
        current_adj[u] |= (1 << v)
        current_adj[v] |= (1 << u)
        
        # Evaluate
        new_tf_count = _count_tf_12(current_adj, masks_12, masks_12_edge_sets)
        
        # Check if we improved
        if new_tf_count < best_tf_count:
            # Check exact score
            new_score = _max_tf_size_exact(current_adj)
            if new_score < best_score:
                best_score = new_score
                best_tf_count = new_tf_count
                best_adj = current_adj[:]
                best_edges = adj_to_edges(current_adj)
                no_improve = 0
                if best_score <= 11:
                    return best_edges
            else:
                # TF count decreased but score didn't (shouldn't happen often)
                # Keep the change if TF count improved
                best_tf_count = new_tf_count
                best_adj = current_adj[:]
                no_improve = 0
        elif new_tf_count == best_tf_count:
            # Neutral move - keep with some probability
            if rng.random() < 0.3:
                best_tf_count = new_tf_count
                best_adj = current_adj[:]
                no_improve = 0
            else:
                # Revert
                current_adj[u] ^= (1 << v)
                current_adj[v] ^= (1 << u)
                no_improve += 1
        else:
            # TF count increased - revert
            current_adj[u] ^= (1 << v)
            current_adj[v] ^= (1 << u)
            no_improve += 1
        
        if no_improve > max_no_improve:
            # Restart with a different random graph
            no_improve = 0
            current_adj = [0] * N
            edges_to_try = ALL_EDGES[:]
            rng.shuffle(edges_to_try)
            for eu, ev in edges_to_try:
                if can_add_edge(current_adj, eu, ev):
                    current_adj[eu] |= (1 << ev)
                    current_adj[ev] |= (1 << eu)
            current_tf_count = _count_tf_12(current_adj, masks_12, masks_12_edge_sets)
            current_score = _max_tf_size_exact(current_adj)
            if current_score < best_score:
                best_score = current_score
                best_tf_count = current_tf_count
                best_adj = current_adj[:]
                best_edges = adj_to_edges(current_adj)
                if best_score <= 11:
                    return best_edges
    
    return best_edges

# EVOLVE-BLOCK-END
>>>>>>> REPLACE
