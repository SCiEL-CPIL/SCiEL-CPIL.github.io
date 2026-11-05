import time
import random
import itertools

_START = time.time()
_DEADLINE = _START + 115.0  # safety margin before 150s kill

def _deadline():
    return time.time() >= _DEADLINE

def _is_regular_mask(adj, mask):
    """True if induced subgraph on mask is regular (all same degree)."""
    first = -1
    m = mask
    while m:
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        m ^= lsb
        deg = (adj[v] & mask).bit_count()
        if first == -1:
            first = deg
        elif deg != first:
            return False
    return True

def _has_regular_subgraph(adj, n, min_size=6):
    """Return True if any subset of size >= min_size induces a regular subgraph."""
    for k in range(min_size, n + 1):
        for comb in itertools.combinations(range(n), k):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            if _is_regular_mask(adj, mask):
                return True
    return False

def _random_graph(n, p=0.5, rng=None):
    rng = rng or random
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _valid(adj, n):
    return not _has_regular_subgraph(adj, n, 6)

def _edges_from_adj(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return edges

def _try_local_search(adj, n, rng, max_iters=300):
    """Local search: flip single edges to eliminate regular subgraphs.
    Uses validity check (early termination) instead of counting."""
    if _valid(adj, n):
        return adj
    
    cur = adj[:]
    no_improve = 0
    for it in range(max_iters):
        if _deadline():
            break
        if _valid(cur, n):
            return cur
        
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j:
            continue
        cur[i] ^= (1 << j)
        cur[j] ^= (1 << i)
        if _valid(cur, n):
            no_improve = 0
        else:
            cur[i] ^= (1 << j)
            cur[j] ^= (1 << i)
            no_improve += 1
        if no_improve > 150:
            # Random restart of a few edges
            for _ in range(3):
                a = rng.randrange(n)
                b = rng.randrange(n)
                if a != b:
                    cur[a] ^= (1 << b)
                    cur[b] ^= (1 << a)
            no_improve = 0
    return cur

def construct():
    rng = random.Random(12345)
    
    # Try from largest n down to find the maximum valid graph
    for target_n in range(22, 5, -1):
        if _deadline():
            break
        found = False
        
        # Strategy 1: Many random graphs
        for attempt in range(300):
            if _deadline():
                break
            p = rng.uniform(0.3, 0.7)
            adj = _random_graph(target_n, p, rng)
            if _valid(adj, target_n):
                found = True
                return {'n': target_n, 'edges': _edges_from_adj(adj, target_n)}
        
        if found:
            break
        
        # Strategy 2: Local search from random graphs
        for attempt in range(30):
            if _deadline():
                break
            p = rng.uniform(0.3, 0.7)
            adj = _random_graph(target_n, p, rng)
            improved = _try_local_search(adj, target_n, rng, max_iters=200)
            if _valid(improved, target_n):
                return {'n': target_n, 'edges': _edges_from_adj(improved, target_n)}
    
    # Fallback: try smaller n
    best_n = 5
    best_adj = [0] * 5
    for target_n in range(16, 5, -1):
        if _deadline():
            break
        for attempt in range(100):
            if _deadline():
                break
            p = rng.uniform(0.3, 0.7)
            adj = _random_graph(target_n, p, rng)
            if _valid(adj, target_n):
                best_n = target_n
                best_adj = adj
                break
        if best_n == target_n:
            break
    
    if _valid(best_adj, best_n):
        return {'n': best_n, 'edges': _edges_from_adj(best_adj, best_n)}
    return {'n': 5, 'edges': []}
