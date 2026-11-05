import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    n = 16
    # Precompute all pairs
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # 1. Generate the Clebsch Graph (known optimal starting point with 40 edges)
    # Vertices 0..15 correspond to the 16 even-parity 5-bit vectors
    vectors = []
    for i in range(32):
        if bin(i).count('1') % 2 == 0:
            vectors.append(i)
            
    clebsch_edges = []
    clebsch_set = set()
    for u_idx in range(16):
        for v_idx in range(u_idx + 1, 16):
            xor = vectors[u_idx] ^ vectors[v_idx]
            if bin(xor).count('1') == 4:
                clebsch_edges.append([u_idx, v_idx])
                clebsch_set.add((u_idx, v_idx))
                
    best_edges = clebsch_edges
    best_count = len(clebsch_edges) # 40
    
    # 2. Efficient 3-colorability check using bitmasks and backtracking
    # We use a bitmask representation for adjacency to make triangle checks and
    # color checks fast.
    
    def is_3_colorable(adj_mask):
        """
        Check if a triangle-free graph (represented by adj_mask list of ints)
        is 3-colorable. Returns True if 3-colorable, False otherwise.
        """
        # Build neighbor lists for faster iteration during backtracking
        neighbors = [[] for _ in range(n)]
        for i in range(n):
            mask = adj_mask[i]
            # Extract set bits
            # Since n=16, we can just loop
            for j in range(n):
                if mask & (1 << j):
                    neighbors[i].append(j)
        
        # Order vertices by degree (highest first) for better pruning
        # This is a static ordering based on current graph structure
        order = sorted(range(n), key=lambda x: -len(neighbors[x]))
        
        colors = [-1] * n
        
        # Additional pruning: If a vertex has 3 neighbors of distinct colors, it's invalid.
        # But standard backtracking with ordering usually suffices for n=16.
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            # Determine forbidden colors
            forbidden = 0
            for u in neighbors[v]:
                if colors[u] != -1:
                    forbidden |= (1 << colors[u])
            
            # Try each color 0, 1, 2
            for c in range(3):
                if not (forbidden & (1 << c)):
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            return False
            
        return backtrack(0)

    # 3. Local Search: Greedy Edge Addition
    # Try to add edges to the current best graph while maintaining triangle-free
    # and non-3-colorable properties.
    
    def can_add_edge(adj_mask, u, v):
        # Check if adding (u,v) creates a triangle.
        # This happens if u and v have a common neighbor.
        common = adj_mask[u] & adj_mask[v]
        return common == 0

    def try_add_edges(adj_mask, current_edges, current_count):
        nonlocal best_edges, best_count
        if time.time() > deadline:
            return current_edges, current_count
            
        # Find all available edges that can be added without creating triangles
        available = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj_mask[u] & (1 << v)):
                    if can_add_edge(adj_mask, u, v):
                        available.append((u, v))
        
        if not available:
            return current_edges, current_count
        
        # Shuffle to break symmetry and explore different paths
        random.shuffle(available)
        
        for u, v in available:
            if time.time() > deadline:
                break
                
            # Tentatively add edge
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            new_edges = current_edges + [[u, v]]
            new_count = current_count + 1
            
            # Check if still 4-chromatic (not 3-colorable)
            if not is_3_colorable(adj_mask):
                # Found a better graph
                if new_count > best_count:
                    best_edges = new_edges
                    best_count = new_count
                # Recurse to try adding more edges
                new_edges, new_count = try_add_edges(adj_mask, new_edges, new_count)
            else:
                # Revert
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
        
        return current_edges, best_count

    # Initialize adjacency mask for Clebsch graph
    adj_mask = [0] * n
    for u, v in clebsch_edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    # Run local search from Clebsch
    _, best_count = try_add_edges(adj_mask, best_edges, best_count)
    
    # 4. Random Restart Search
    # If local search from Clebsch didn't find much, try random constructions.
    # Strategy: Start with random triangle-free graph, check 4-chromaticity,
    # then try to add edges.
    
    # Pre-compute candidate edges not in Clebsch for faster iteration? 
    # Actually, just use all_pairs and check validity on the fly.
    
    for iteration in range(5000):
        if time.time() > deadline:
            break
            
        random.shuffle(all_pairs)
        adj_mask = [0] * n
        edges = []
        
        # Greedy construction of a random triangle-free graph
        for u, v in all_pairs:
            if time.time() > deadline:
                break
            if can_add_edge(adj_mask, u, v):
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                edges.append([u, v])
        
        # Check if this random graph is 4-chromatic
        if not is_3_colorable(adj_mask):
            if len(edges) > best_count:
                best_edges = edges
                best_count = len(edges)
                # Try to improve this random find
                _, best_count = try_add_edges(adj_mask, edges, len(edges))
            # Even if not better than best, we might try to add edges if it's close?
            # No, just check if we can add edges to it if it's already good.
            elif len(edges) >= best_count - 5:
                 # If it's close, maybe it can be extended?
                 _, best_count = try_add_edges(adj_mask, edges, len(edges))

    return best_edges
