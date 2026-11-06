import time
import random
import itertools

N = 16
ALL_BITS = (1 << N) - 1

def _groetzsch_edges():
    """
    Grötzsch graph (Mycielski of C5).
    11 vertices, 20 edges.
    """
    edges = []
    # Original C5: 0-1, 1-2, 2-3, 3-4, 4-0
    for i in range(5):
        edges.append([i, (i+1)%5])
        
    # Copies 5-9 correspond to 0-4.
    # Center 10.
    # Edges between original and copies:
    # If (u,v) is an edge in C5, then (v, u') and (u, v') are edges in Mycielski.
    orig_edges = [[0,1], [1,2], [2,3], [3,4], [4,0]]
    for u, v in orig_edges:
        edges.append([v, 5+u])
        edges.append([u, 5+v])
        
    # Center 10 connected to all copies 5-9
    for i in range(5):
        edges.append([10, 5+i])
        
    return edges

def _adj_from_edges(edge_list):
    adj = [0] * N
    for u, v in edge_list:
        if u < N and v < N:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    return adj

def _edge_count(adj):
    return sum(bin(a).count("1") for a in adj) // 2

def _is_triangle_free_add(adj, u, v):
    # Check if adding (u,v) creates a triangle.
    return (adj[u] & adj[v]) == 0

def _is_3_colorable_fast(adj):
    """
    Fast bitmask-based backtracking to check if graph is 3-colorable.
    Uses vertex ordering by degree and forward checking.
    """
    colors = [0] * N  # 0=uncolored, 1/2/3 = colors
    # Order vertices by degree descending
    degs = [bin(adj[x]).count('1') for x in range(N)]
    order = sorted(range(N), key=lambda x: degs[x], reverse=True)
    
    # Precompute neighbor masks for each vertex
    nbr_masks = list(adj)
    
    def backtrack(i):
        if i == N:
            return True
        v = order[i]
        # Get available colors based on colored neighbors
        forbidden = 0
        m = nbr_masks[v]
        while m:
            lb = m & (-m)
            u = lb.bit_length() - 1
            m ^= lb
            if colors[u] != 0:
                forbidden |= (1 << (colors[u] - 1))
        
        available = 7 & ~forbidden  # 3 bits for 3 colors
        if available == 0:
            return False
        
        # Try available colors
        while available:
            lb = available & (-available)
            c = lb.bit_length()  # 1, 2, or 3
            available ^= lb
            colors[v] = c
            if backtrack(i + 1):
                return True
            colors[v] = 0
        return False
        
    return backtrack(0)

def _greedy_fill(adj, deadline, rng):
    """
    Greedily add edges to adj, respecting triangle-freeness.
    Prioritize edges that connect low-degree vertices first.
    """
    candidates = []
    for u in range(N):
        for v in range(u + 1, N):
            if not ((adj[u] >> v) & 1):
                du = bin(adj[u]).count('1')
                dv = bin(adj[v]).count('1')
                # Lower sum of degrees means more likely to be addable
                candidates.append((du + dv, u, v))
    
    rng.shuffle(candidates)
    candidates.sort(key=lambda x: x[0])  # Sort by degree sum
    
    for _, u, v in candidates:
        if time.time() > deadline:
            break
        if _is_triangle_free_add(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
    return adj

def _local_improve(adj, deadline, rng):
    """
    Local search to maximize edges with better move generation.
    """
    improved = True
    best_cnt = _edge_count(adj)
    
    while improved and time.time() < deadline:
        improved = False
        current_cnt = _edge_count(adj)
        
        # 1. Try to add any missing edge
        missing = []
        for u in range(N):
            for v in range(u + 1, N):
                if not ((adj[u] >> v) & 1):
                    missing.append((u, v))
        rng.shuffle(missing)
        
        for u, v in missing:
            if _is_triangle_free_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                improved = True
                break
        
        if not improved:
            # 2. Try random edge removals followed by greedy refill
            existing = []
            for u in range(N):
                for v in range(u + 1, N):
                    if (adj[u] >> v) & 1:
                        existing.append((u, v))
            
            if not existing:
                break

            best_adj = None
            best_local_cnt = current_cnt
            
            # Try more removals
            num_tries = min(20, len(existing))
            for _ in range(num_tries):
                if time.time() > deadline:
                    break
                rem_u, rem_v = rng.choice(existing)
                
                # Copy adj
                adj_copy = [a for a in adj]
                adj_copy[rem_u] &= ~(1 << rem_v)
                adj_copy[rem_v] &= ~(1 << rem_u)
                
                # Greedy fill
                adj_copy = _greedy_fill(adj_copy, deadline, rng)
                cnt = _edge_count(adj_copy)
                
                if cnt > best_local_cnt:
                    best_local_cnt = cnt
                    best_adj = adj_copy
            
            if best_adj and best_local_cnt > best_cnt:
                adj = best_adj
                best_cnt = best_local_cnt
                improved = True

    return adj

def _partition_search(deadline, rng):
    """
    Structural approach: Partition 16 vertices into S1, S2, S3, {v}.
    Maximize edges between sets subject to triangle-freeness.
    """
    best_adj = None
    best_cnt = -1
    
    # We need to partition 16 vertices into 4 groups: S1, S2, S3, {v}
    # where |{v}| = 1.
    # Total vertices = 16.
    # Let sizes be s1, s2, s3, 1.
    # s1 + s2 + s3 = 15.
    # To maximize edges, we want the partition to be as balanced as possible?
    # Or maybe unbalanced?
    # Triangle-free constraint:
    # Edges within Si are forbidden (independent sets).
    # Edges between Si and Sj must not create triangles.
    # A triangle involves one vertex from each of 3 sets? No, if S1, S2, S3 are independent,
    # a triangle (u,v,w) must have u in Si, v in Sj, w in Sk.
    # If i,j,k are distinct, then u-v, v-w, w-u are edges.
    # This implies a C3 in the "block graph" if we consider connections.
    # Actually, if we have edges between S1-S2, S2-S3, S3-S1, we must avoid triangles.
    # A triangle is formed if there exist u in S1, v in S2, w in S3 such that
    # u-v, v-w, w-u are all edges.
    # This is equivalent to saying the bipartite graphs between pairs must not form
    # a "triangle" in the tripartite graph.
    
    # Let's try specific size distributions.
    # 15 vertices to split into 3 sets.
    # Balanced: 5, 5, 5.
    # Unbalanced: 7, 4, 4. 8, 3, 4. etc.
    
    size_dists = [
        (5, 5, 5),
        (6, 5, 4),
        (7, 4, 4),
        (6, 6, 3),
        (7, 5, 3),
        (8, 4, 3),
        (7, 6, 2),
        (8, 5, 2),
        (9, 4, 2),
        (8, 6, 1),
        (9, 5, 1),
        (10, 4, 1),
        (9, 6, 0), # One set empty? Then it's bipartite + 1 vertex.
        (10, 5, 0),
    ]
    
    for s1, s2, s3 in size_dists:
        if s1 + s2 + s3 != 15:
            continue
            
        # Generate a random partition of 15 vertices into sets of these sizes
        # and 1 vertex v.
        # To save time, we will try a few random partitions for each size distribution.
        num_partitions = 5
        for _ in range(num_partitions):
            if time.time() > deadline:
                break
                
            # Shuffle vertices
            verts = list(range(15))
            rng.shuffle(verts)
            
            S1 = verts[:s1]
            S2 = verts[s1:s1+s2]
            S3 = verts[s1+s2:s1+s2+s3]
            v = 15 # The 16th vertex
            
            # Build adjacency
            adj = [0] * N
            
            # Add edges between S1, S2, S3
            # We want to maximize edges.
            # Start with all possible edges, then remove those that create triangles.
            # Or add greedily?
            # Greedy is safer.
            
            # Order of addition:
            # Maybe add S1-S2, S2-S3, S3-S1?
            # Let's try to add all possible edges and check triangle-freeness.
            # But we must ensure NO triangles.
            
            # A triangle is (u in S1, v in S2, w in S3) with u-v, v-w, w-u edges.
            # This is a 3-cycle in the tripartite graph.
            
            # Strategy:
            # 1. Add all edges between S1 and S2? No, that might be too many.
            # Let's try to add edges one by one, checking for triangles.
            # To maximize, we should try to add as many as possible.
            
            # Let's use a randomized greedy approach for the partition structure.
            # Candidates are all pairs (u,v) where u in Si, v in Sj, i != j.
            # Plus edges involving v?
            # v can connect to anyone?
            # If v connects to u in S1, and u connects to w in S2, and w connects to z in S3...
            # Triangles involving v: (v, u, w) where u-w is an edge.
            # So if v-u and v-w are edges, u-w cannot be an edge.
            
            # This is complex. Let's stick to the main partition edges first.
            
            candidates = []
            # Pairs within S1, S2, S3 are forbidden (independent sets)
            # Pairs between different sets are candidates.
            sets = [S1, S2, S3]
            for i in range(3):
                for j in range(i+1, 3):
                    for u in sets[i]:
                        for v in sets[j]:
                            candidates.append((u, v))
                            
            # Add v to candidates?
            # v is connected to S1, S2, S3.
            for u in S1:
                candidates.append((v, u))
            for u in S2:
                candidates.append((v, u))
            for u in S3:
                candidates.append((v, u))
                
            rng.shuffle(candidates)
            
            # Greedy add
            for u, w in candidates:
                if time.time() > deadline:
                    break
                if _is_triangle_free_add(adj, u, w):
                    adj[u] |= (1 << w)
                    adj[w] |= (1 << u)
                    
            # Local improve
            adj = _local_improve(adj, deadline, rng)
            
            cnt = _edge_count(adj)
            if cnt > best_cnt:
                if not _is_3_colorable_fast(adj):
                    best_cnt = cnt
                    best_adj = adj
                    
    return best_adj, best_cnt

def construct():
    start = time.time()
    deadline = start + 115.0  # Safety margin
    
    base_edges = _groetzsch_edges()
    
    best_adj = None
    best_cnt = -1
    
    rng = random.Random(42)
    
    # Strategy 1: Partition Search
    part_adj, part_cnt = _partition_search(deadline, rng)
    if part_adj:
        best_adj = part_adj
        best_cnt = part_cnt
        
    # Strategy 2: Grötzsch based
    strategies = []
    
    # Strategy 1: Standard Grötzsch base
    strategies.append(("standard", _adj_from_edges(base_edges)))
    
    # Strategy 2: Grötzsch with some edges removed for diversity
    for _ in range(3):
        adj_variant = _adj_from_edges(base_edges)
        # Remove 1-2 random edges
        for _ in range(rng.randint(1, 2)):
            existing = []
            for u in range(N):
                for v in range(u + 1, N):
                    if (adj_variant[u] >> v) & 1:
                        existing.append((u, v))
            if existing:
                ru, rv = rng.choice(existing)
                adj_variant[ru] &= ~(1 << rv)
                adj_variant[rv] &= ~(1 << ru)
        strategies.append(("variant", adj_variant))
    
    # Strategy 3: Start from scratch with random triangle-free edges
    for _ in range(2):
        adj_random = [0] * N
        # Add random edges while maintaining triangle-free
        all_pairs = [(u, v) for u in range(N) for v in range(u+1, N)]
        rng.shuffle(all_pairs)
        for u, v in all_pairs:
            if _is_triangle_free_add(adj_random, u, v):
                if rng.random() < 0.3:  # Only add 30% of possible edges
                    adj_random[u] |= (1 << v)
                    adj_random[v] |= (1 << u)
        # Ensure it's not 3-colorable by checking; if it is, skip
        if not _is_3_colorable_fast(adj_random):
            strategies.append(("random", adj_random))
    
    # Process each strategy
    for strat_name, adj in strategies:
        if time.time() > deadline - 5.0:
            break
        
        # Greedy fill
        adj = _greedy_fill(adj, deadline, rng)
        
        # Local improve
        adj = _local_improve(adj, deadline, rng)
        
        cnt = _edge_count(adj)
        
        # Check if it's still 4-chromatic
        if cnt > best_cnt:
            if not _is_3_colorable_fast(adj):
                best_cnt = cnt
                best_adj = adj
    
    # Fallback
    if best_adj is None:
        best_adj = _adj_from_edges(base_edges)
    
    # Final Verification
    if _is_3_colorable_fast(best_adj):
        best_adj = _adj_from_edges(base_edges)
    
    # Convert back to edge list
    out = []
    for u in range(N):
        m = best_adj[u]
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            m ^= lb
            if v > u:
                out.append([u, v])
                
    return out
