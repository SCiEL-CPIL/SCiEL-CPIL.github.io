import time
import random
import itertools

def _precompute_tri_free(adj_mask, n):
    """Precompute which subsets are triangle-free using DP with bitmasks.
    adj_mask[i] is a bitmask of neighbors of vertex i.
    Returns a list of length 2^n where tri_free[mask] is True if the subset is triangle-free."""
    size = 1 << n
    tri_free = [False] * size
    tri_free[0] = True
    
    # Precompute: for each mask, is it an independent set?
    # indep[mask] = True if no two vertices in mask are adjacent
    indep = [False] * size
    indep[0] = True
    for mask in range(1, size):
        # Get lowest set bit
        lb = mask & (-mask)
        v = lb.bit_length() - 1
        prev = mask ^ lb
        # mask is independent iff prev is independent AND v has no neighbors in prev
        indep[mask] = indep[prev] and (adj_mask[v] & prev) == 0
    
    # DP for triangle-free
    for mask in range(1, size):
        lb = mask & (-mask)
        v = lb.bit_length() - 1
        prev = mask ^ lb
        if not tri_free[prev]:
            tri_free[mask] = False
        else:
            # Check if v forms a triangle with any pair in prev
            # v forms triangle with (a,b) iff adj[v][a] and adj[v][b] and adj[a][b]
            # i.e., the neighbors of v in prev must be an independent set
            neigh_in_prev = adj_mask[v] & prev
            tri_free[mask] = indep[neigh_in_prev]
    
    return tri_free

def _max_tf_size(adj_mask, n, tri_free):
    """Find the largest triangle-free subset size."""
    size = 1 << n
    best = 0
    for mask in range(size):
        if tri_free[mask]:
            c = mask.bit_count()
            if c > best:
                best = c
    return best

def _is_k4_free_mask(adj_mask, n):
    """Check if graph is K4-free using bitmasks."""
    for a in range(n):
        for b in range(a+1, n):
            if not (adj_mask[a] >> b) & 1:
                continue
            # Common neighbors of a and b
            common = adj_mask[a] & adj_mask[b]
            # For each pair of common neighbors c,d, check if they're adjacent
            # If any c,d in common with adj[c][d], then a,b,c,d form K4
            # Actually: K4 = a,b,c,d all connected. a-b connected (checked).
            # Need c,d in common, and c-d connected, and a-c, a-d, b-c, b-d (all true since c,d in common)
            # So we need: does the subgraph induced by 'common' have any edge?
            # i.e., is 'common' NOT an independent set?
            # Check: for each c in common, is (adj_mask[c] & common) != 0?
            c_bits = common
            while c_bits:
                lb = c_bits & (-c_bits)
                c = lb.bit_length() - 1
                if adj_mask[c] & common:
                    return False
                c_bits ^= lb
    return True

def _compute_score(adj_mask, n):
    """Compute the largest triangle-free induced subgraph size."""
    tri_free = _precompute_tri_free(adj_mask, n)
    return _max_tf_size(adj_mask, n, tri_free)

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin for 150s limit
    n = 18
    size = 1 << n
    
    def time_ok():
        return time.time() < deadline
    
    best_score = 18  # worst case
    best_adj_mask = [0] * n
    
    def evaluate(adj_mask):
        """Return (score, adj_mask) if K4-free, else None."""
        if not _is_k4_free_mask(adj_mask, n):
            return None
        score = _compute_score(adj_mask, n)
        return (score, adj_mask)
    
    # Construction 1: K6,6,6
    adj1 = [0] * n
    for i in range(6):
        for j in range(6, 12):
            adj1[i] |= (1 << j)
            adj1[j] |= (1 << i)
        for j in range(12, 18):
            adj1[i] |= (1 << j)
            adj1[j] |= (1 << i)
    for i in range(6, 12):
        for j in range(12, 18):
            adj1[i] |= (1 << j)
            adj1[j] |= (1 << i)
    
    result = evaluate(adj1)
    if result:
        best_score, best_adj_mask = result
    
    # Construction 2: Random dense K4-free graphs
    random.seed(42)
    
    for trial in range(200):
        if not time_ok():
            break
        
        # Start with empty graph, add random edges while maintaining K4-free
        adj = [0] * n
        edges = list(itertools.combinations(range(n), 2))
        random.shuffle(edges)
        
        for u, v in edges:
            if not time_ok():
                break
            # Try adding edge (u,v)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if not _is_k4_free_mask(adj, n):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        result = evaluate(adj)
        if result:
            score, _ = result
            if score < best_score:
                best_score = score
                best_adj_mask = adj[:]
    
    # Construction 3: Local search on best graph
    # Try adding edges to the best graph
    if time_ok():
        adj = best_adj_mask[:]
        edges = list(itertools.combinations(range(n), 2))
        random.shuffle(edges)
        for u, v in edges:
            if not time_ok():
                break
            if (adj[u] >> v) & 1:
                continue  # edge already exists
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if _is_k4_free_mask(adj, n):
                result = evaluate(adj)
                if result:
                    score, _ = result
                    if score < best_score:
                        best_score = score
                        best_adj_mask = adj[:]
                    # Keep the edge even if score doesn't improve (might help later)
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
    
    # Construction 4: More random trials with different seeds
    for seed in [1, 7, 13, 99, 123]:
        if not time_ok():
            break
        random.seed(seed)
        for trial in range(50):
            if not time_ok():
                break
            adj = [0] * n
            edges = list(itertools.combinations(range(n), 2))
            random.shuffle(edges)
            for u, v in edges:
                if not time_ok():
                    break
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if not _is_k4_free_mask(adj, n):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
            result = evaluate(adj)
            if result:
                score, _ = result
                if score < best_score:
                    best_score = score
                    best_adj_mask = adj[:]
    
    # Convert to edge list
    best_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if (best_adj_mask[u] >> v) & 1:
                best_edges.append([u, v])
    
    return best_edges
