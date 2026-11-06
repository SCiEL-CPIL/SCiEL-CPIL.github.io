import time

DEADLINE = 140.0

def _construct_andrasfai(k):
    """Construct the Andrasfai graph A_k on k^2 vertices."""
    n = k * k
    adj = [0] * n
    for i in range(n):
        x1 = i // k
        y1 = i % k
        for j in range(i + 1, n):
            x2 = j // k
            y2 = j % k
            if x1 != x2:
                if (y2 - y1) % k == (x1 * x2) % k:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
    return adj, n

def _has_triangle_fast(adj, n):
    """Fast triangle check using bitsets."""
    for i in range(n):
        ai = adj[i]
        j = ai & ~((1 << (i + 1)) - 1)
        while j:
            lb = j & (-j)
            jj = lb.bit_length() - 1
            common = ai & adj[jj] & ~((1 << (jj + 1)) - 1)
            if common:
                return True
            j ^= lb
    return False

def _alpha_exact_bnb(adj, n, limit=10):
    """Exact max independent set size via branch and bound. Returns >= limit if alpha >= limit."""
    best = [0]
    
    def bnb(chosen_count, candidates):
        if chosen_count + candidates.bit_count() <= best[0]:
            return
        if chosen_count >= limit:
            best[0] = chosen_count
            return
        if candidates == 0:
            if chosen_count > best[0]:
                best[0] = chosen_count
            return
        
        # Heuristic: pick vertex with highest degree in candidates
        # For small n, simple lowest index is fine, but degree helps.
        # However, calculating degree is O(n). 
        # For n=40, it's fast.
        # Let's stick to simple branch and bound with bit_count pruning.
        
        lb = candidates & (-candidates)
        v = lb.bit_length() - 1
        rest = candidates ^ lb
        
        # Branch: exclude v
        bnb(chosen_count, rest)
        
        # Branch: include v
        new_cands = rest & ~adj[v]
        bnb(chosen_count + 1, new_cands)
    
    bnb(0, (1 << n) - 1)
    return best[0]

def _is_valid(adj, n):
    """Check triangle-free and alpha < 10."""
    if _has_triangle_fast(adj, n):
        return False
    if _alpha_exact_bnb(adj, n, limit=10) >= 10:
        return False
    return True

def construct():
    start_time = time.time()
    
    # Use Andrasfai graph A_9 (81 vertices)
    # It is triangle-free.
    # We take the first 40 vertices.
    # The independence number of the induced subgraph will be <= alpha(A_9).
    # If alpha(A_9) < 10, we are good.
    # Even if alpha(A_9) >= 10, the subgraph might have alpha < 10.
    
    k = 9
    adj_full, n_full = _construct_andrasfai(k)
    
    n_target = 40
    if n_full >= n_target:
        adj = [0] * n_target
        mask = (1 << n_target) - 1
        for i in range(n_target):
            adj[i] = adj_full[i] & mask
        
        if _is_valid(adj, n_target):
            edges = []
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    if adj[i] & (1 << j):
                        edges.append([i, j])
            return {'n': n_target, 'edges': edges}
    
    # Fallback: Try n=39
    for n_try in range(39, 20, -1):
        if time.time() > start_time + DEADLINE - 5:
            break
        
        # Try subgraph of A_9 for smaller n
        if n_full >= n_try:
            adj = [0] * n_try
            mask = (1 << n_try) - 1
            for i in range(n_try):
                adj[i] = adj_full[i] & mask
            
            if _is_valid(adj, n_try):
                edges = []
                for i in range(n_try):
                    for j in range(i + 1, n_try):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n_try, 'edges': edges}
        
        # If Andrasfai subgraph fails, try random
        # (Unlikely to be needed if A_9 works)
        
    return {'n': 0, 'edges': []}
