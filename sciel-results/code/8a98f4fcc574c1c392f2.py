"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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
    
    return tri_free, indep

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
    tri_free, _ = _precompute_tri_free(adj_mask, n)
    return _max_tf_size(adj_mask, n, tri_free)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    n = 18
    size = 1 << n
    
    def time_ok():
        return time.time() < deadline
    
    best_score = 18  # Worst case (empty graph)
    best_adj_mask = [0] * n
    
    def evaluate(adj_mask):
        """Return (score, adj_mask) if K4-free, else None."""
        if not _is_k4_free_mask(adj_mask, n):
            return None
        score = _compute_score(adj_mask, n)
        return (score, adj_mask)
    
    # Construction 1: Turan Graph T(18, 3) = K6,6,6
    # This is a known dense K4-free graph.
    adj1 = [0] * n
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(3):
        for j in range(i+1, 3):
            for u in parts[i]:
                for v in parts[j]:
                    adj1[u] |= (1 << v)
                    adj1[v] |= (1 << u)
    
    result = evaluate(adj1)
    if result:
        best_score, best_adj_mask = result

    # Construction 2: Random Dense K4-Free Graphs
    # Strategy: Start with a random graph with high edge probability, 
    # then remove edges that are part of any K4.
    random.seed(42)
    
    # Helper to remove K4s
    def remove_k4s(adj):
        changed = True
        while changed:
            changed = False
            # Find all K4s
            for a in range(n):
                for b in range(a+1, n):
                    if not (adj[a] >> b) & 1:
                        continue
                    common = adj[a] & adj[b]
                    c_bits = common
                    while c_bits:
                        lb = c_bits & (-c_bits)
                        c = lb.bit_length() - 1
                        # Check if c has any neighbor in common (excluding itself, but common doesn't have itself)
                        # If adj[c] & common has any bit set, there is a K4
                        # Actually, we just need to check if there is an edge between c and another vertex in common
                        # If we remove edge (a,b), does it break all K4s containing (a,b)?
                        # It's safer to just check if the graph is K4-free after removal.
                        # But that's slow.
                        # Let's just pick a random edge from a K4 and remove it.
                        # For efficiency in the loop, we'll just check K4-freeness of the whole graph.
                        # To speed up, we can just check if common contains an edge.
                        if adj[c] & common:
                             # Found a K4 involving a,b,c. Remove edge (a,b).
                             adj[a] &= ~(1 << b)
                             adj[b] &= ~(1 << a)
                             changed = True
                             c_bits = 0 # Break inner loop
                             break
                        c_bits ^= lb
                    if not c_bits: # If we broke due to finding K4
                        break
                if changed:
                    break
            if changed:
                break
        return adj

    # Generate random dense graphs
    for trial in range(500):
        if not time_ok():
            break
        
        # Random graph with p=0.5
        adj = [0] * n
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < 0.5:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Prune K4s
        adj = remove_k4s(adj)
        
        result = evaluate(adj)
        if result:
            score, _ = result
            if score < best_score:
                best_score = score
                best_adj_mask = adj[:]

    # Construction 3: Local Search on Best Graph
    # Try 1-flip moves: add an edge if it doesn't create K4 and improves score.
    # Try 2-flip moves: swap an edge.
    
    # First, try to add edges to the best graph found so far
    if time_ok():
        adj = best_adj_mask[:]
        edges = list(itertools.combinations(range(n), 2))
        random.shuffle(edges)
        
        # Try adding missing edges
        for u, v in edges:
            if not time_ok():
                break
            if (adj[u] >> v) & 1:
                continue  # Edge already exists
            
            # Check if adding (u,v) creates a K4
            # It creates a K4 if there exists a pair (a,b) such that u-v-a-b is a K4?
            # No, it creates a K4 if there is a triangle (u, a, b) or (v, a, b) 
            # such that a-b is an edge and a,b are connected to u/v appropriately.
            # Simpler: Just add it, check K4-free, then check score.
            
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if _is_k4_free_mask(adj, n):
                result = evaluate(adj)
                if result:
                    score, _ = result
                    if score < best_score:
                        best_score = score
                        best_adj_mask = adj[:]
                    # Keep the edge even if score doesn't improve? 
                    # Adding edges generally makes it harder to be triangle-free, so it should help or stay same.
                    # But it might block future better moves.
                    # Let's keep it if it doesn't hurt.
            else:
                # Revert
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)

    # Second, try random restarts with local search
    for seed in [1, 7, 13, 99, 123, 456, 789]:
        if not time_ok():
            break
        random.seed(seed)
        for trial in range(50):
            if not time_ok():
                break
            
            # Start with a random dense graph
            adj = [0] * n
            for u in range(n):
                for v in range(u+1, n):
                    if random.random() < 0.6: # Higher density
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            adj = remove_k4s(adj)
            
            result = evaluate(adj)
            if not result:
                continue
            
            score, _ = result
            if score < best_score:
                best_score = score
                best_adj_mask = adj[:]
            
            # Local search: try adding edges
            edges = list(itertools.combinations(range(n), 2))
            random.shuffle(edges)
            improved = True
            while improved and time_ok():
                improved = False
                for u, v in edges:
                    if not time_ok():
                        break
                    if (adj[u] >> v) & 1:
                        continue
                    
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
                    if _is_k4_free_mask(adj, n):
                        res = evaluate(adj)
                        if res:
                            new_score, _ = res
                            if new_score < score:
                                score = new_score
                                improved = True
                                if score < best_score:
                                    best_score = score
                                    best_adj_mask = adj[:]
                            # Keep edge if it doesn't worsen score?
                            # Adding edges reduces triangle-free sets.
                            # So new_score <= score usually.
                            # If new_score == score, we keep it to make graph denser.
                            # If new_score > score, we revert.
                        else:
                            # Should not happen if K4-free check passed
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)

    # Convert to edge list
    best_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if (best_adj_mask[u] >> v) & 1:
                best_edges.append([u, v])
    
    return best_edges
# EVOLVE-BLOCK-END
