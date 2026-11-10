import time
import random
import itertools

def _precompute_tri_free(adj_mask, n):
    """Precompute which subsets are triangle-free using DP with bitmasks."""
    size = 1 << n
    tri_free = [False] * size
    tri_free[0] = True
    
    indep = [False] * size
    indep[0] = True
    for mask in range(1, size):
        lb = mask & (-mask)
        v = lb.bit_length() - 1
        prev = mask ^ lb
        indep[mask] = indep[prev] and (adj_mask[v] & prev) == 0
    
    for mask in range(1, size):
        lb = mask & (-mask)
        v = lb.bit_length() - 1
        prev = mask ^ lb
        if not tri_free[prev]:
            tri_free[mask] = False
        else:
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
            common = adj_mask[a] & adj_mask[b]
            c_bits = common
            while c_bits:
                lb = c_bits & (-c_bits)
                c = lb.bit_length() - 1
                if adj_mask[c] & common:
                    return False
                c_bits ^= lb
    return True

def _is_k4_free_increm(adj_mask, n, u, v):
    """Check if adding edge (u,v) creates a K4. Only check K4s involving (u,v)."""
    # A K4 containing edge (u,v) needs two other vertices c,d such that
    # all of u-v, u-c, u-d, v-c, v-d, c-d are edges
    # c and d must be common neighbors of u and v
    common = adj_mask[u] & adj_mask[v]
    # Remove u and v from common (they shouldn't be there anyway)
    common &= ~(1 << u)
    common &= ~(1 << v)
    # Check if any two vertices in common are adjacent
    c_bits = common
    while c_bits:
        lb = c_bits & (-c_bits)
        c = lb.bit_length() - 1
        # Check if c is adjacent to any other vertex in common
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
    deadline = start_time + 140  # safety margin for 150s limit
    n = 18
    
    def time_ok():
        return time.time() < deadline
    
    best_score = 18
    best_adj_mask = [0] * n
    
    def evaluate(adj_mask):
        if not _is_k4_free_mask(adj_mask, n):
            return None
        score = _compute_score(adj_mask, n)
        return (score, adj_mask)
    
    # Construction 1: K6,6,6 as baseline
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
    
    # Construction 2: Random K4-free graphs with incremental K4 check
    random.seed(42)
    
    for trial in range(500):
        if not time_ok():
            break
        
        adj = [0] * n
        edges = list(itertools.combinations(range(n), 2))
        random.shuffle(edges)
        
        for u, v in edges:
            if not time_ok():
                break
            # Use incremental K4 check
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if not _is_k4_free_increm(adj, n, u, v):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        result = evaluate(adj)
        if result:
            score, _ = result
            if score < best_score:
                best_score = score
                best_adj_mask = adj[:]
    
    # Construction 3: Local search - add edges to best graph
    if time_ok():
        adj = best_adj_mask[:]
        missing_edges = []
        for u in range(n):
            for v in range(u+1, n):
                if not (adj[u] >> v) & 1:
                    missing_edges.append((u, v))
        random.shuffle(missing_edges)
        
        for u, v in missing_edges:
            if not time_ok():
                break
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if _is_k4_free_increm(adj, n, u, v):
                result = evaluate(adj)
                if result:
                    score, _ = result
                    if score < best_score:
                        best_score = score
                        best_adj_mask = adj[:]
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
    
    # Construction 4: Simulated annealing on edge set
    if time_ok():
        adj = best_adj_mask[:]
        current_score = best_score
        temp = 5.0
        cooling = 0.995
        
        all_edges = list(itertools.combinations(range(n), 2))
        random.shuffle(all_edges)
        
        sa_iter = 0
        while time_ok() and sa_iter < 5000:
            sa_iter += 1
            # Pick a random edge to flip
            u, v = random.choice(all_edges)
            has_edge = (adj[u] >> v) & 1
            
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            
            if not _is_k4_free_mask(adj, n):
                # Revert
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                temp *= cooling
                continue
            
            new_score = _compute_score(adj, n)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj_mask = adj[:]
            else:
                # Accept with probability exp(-(new_score - current_score) / temp)
                delta = new_score - current_score
                if delta > 0:
                    accept_prob = 2.718281828 ** (-delta / max(temp, 0.01))
                    if random.random() > accept_prob:
                        # Revert
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                # If delta <= 0, already accepted above
            
            temp *= cooling
    
    # Construction 5: More random trials with different seeds
    for seed in [1, 7, 13, 99, 123, 456, 789, 1000]:
        if not time_ok():
            break
        random.seed(seed)
        for trial in range(100):
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
                if not _is_k4_free_increm(adj, n, u, v):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
            result = evaluate(adj)
            if result:
                score, _ = result
                if score < best_score:
                    best_score = score
                    best_adj_mask = adj[:]
    
    # Construction 6: Final local search from best
    if time_ok():
        adj = best_adj_mask[:]
        missing_edges = []
        for u in range(n):
            for v in range(u+1, n):
                if not (adj[u] >> v) & 1:
                    missing_edges.append((u, v))
        random.shuffle(missing_edges)
        
        for u, v in missing_edges:
            if not time_ok():
                break
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if _is_k4_free_increm(adj, n, u, v):
                result = evaluate(adj)
                if result:
                    score, _ = result
                    if score < best_score:
                        best_score = score
                        best_adj_mask = adj[:]
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
