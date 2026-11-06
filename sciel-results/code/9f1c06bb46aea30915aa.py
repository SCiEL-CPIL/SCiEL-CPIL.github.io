import time
import random

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

def construct():
    start = time.time()
    deadline = start + 115.0  # Safety margin
    
    base_edges = _groetzsch_edges()
    
    best_adj = None
    best_cnt = -1
    
    rng = random.Random(42)
    
    # Try multiple different strategies
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
