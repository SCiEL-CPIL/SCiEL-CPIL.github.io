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

def _count_regular_subgraphs(adj, n, min_size=6):
    """Count number of regular induced subgraphs of size >= min_size."""
    cnt = 0
    for k in range(min_size, n + 1):
        for comb in itertools.combinations(range(n), k):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            if _is_regular_mask(adj, mask):
                cnt += 1
    return cnt

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

def _try_local_search(adj, n, rng, max_iters=5000):
    """Local search: flip single edges to eliminate regular subgraphs."""
    if _deadline():
        return adj
    best = adj[:]
    best_cnt = _count_regular_subgraphs(best, n, 6)
    if best_cnt == 0:
        return best
    
    cur = best[:]
    cur_cnt = best_cnt
    no_improve = 0
    for it in range(max_iters):
        if _deadline():
            break
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j:
            continue
        bit_i = 1 << i
        bit_j = 1 << j
        cur[i] ^= bit_j
        cur[j] ^= bit_i
        new_cnt = _count_regular_subgraphs(cur, n, 6)
        if new_cnt < cur_cnt:
            cur_cnt = new_cnt
            if cur_cnt < best_cnt:
                best = cur[:]
                best_cnt = cur_cnt
            no_improve = 0
        else:
            cur[i] ^= bit_j
            cur[j] ^= bit_i
            no_improve += 1
        if no_improve > 300:
            for _ in range(3):
                a = rng.randrange(n)
                b = rng.randrange(n)
                if a == b:
                    continue
                cur[a] ^= (1 << b)
                cur[b] ^= (1 << a)
            cur_cnt = _count_regular_subgraphs(cur, n, 6)
            no_improve = 0
    return best

def _edges_from_adj(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return edges

def _construct_cyclic_graph(n, rng):
    """Construct a graph based on cyclic structure to break regularity."""
    adj = [0] * n
    # Use a random permutation to create asymmetric structure
    perm = list(range(n))
    rng.shuffle(perm)
    # Add edges based on distance in the permuted order with random thresholds
    for i in range(n):
        for j in range(i + 1, n):
            dist = abs(perm.index(i) - perm.index(j))
            # Use a non-uniform probability based on distance
            p = 0.3 + 0.4 * (dist / n) if dist % 3 != 0 else 0.2
            if rng.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def construct():
    rng = random.Random(12345)
    best_n = 5
    best_adj = [0] * 5
    best_edges = []

    # Try increasing n from 6 to 22
    for target_n in range(6, 23):
        if _deadline():
            break
        found = False
        
        # Strategy 1: Random graphs with various densities
        for attempt in range(50):
            if _deadline():
                break
            p = rng.uniform(0.3, 0.7)
            adj = _random_graph(target_n, p, rng)
            if _valid(adj, target_n):
                found = True
                best_n = target_n
                best_adj = adj
                best_edges = _edges_from_adj(adj, target_n)
                break
            else:
                improved = _try_local_search(adj, target_n, rng, max_iters=1000)
                if _valid(improved, target_n):
                    found = True
                    best_n = target_n
                    best_adj = improved
                    best_edges = _edges_from_adj(improved, target_n)
                    break
        
        if found:
            continue
        
        # Strategy 2: Cyclic-based constructions
        for attempt in range(30):
            if _deadline():
                break
            adj = _construct_cyclic_graph(target_n, rng)
            if _valid(adj, target_n):
                found = True
                best_n = target_n
                best_adj = adj
                best_edges = _edges_from_adj(adj, target_n)
                break
            else:
                improved = _try_local_search(adj, target_n, rng, max_iters=1000)
                if _valid(improved, target_n):
                    found = True
                    best_n = target_n
                    best_adj = improved
                    best_edges = _edges_from_adj(improved, target_n)
                    break
        
        if found:
            continue
        
        # Strategy 3: Start from a smaller valid graph and try to extend
        if best_n >= 6 and best_n < target_n:
            for attempt in range(20):
                if _deadline():
                    break
                # Try to add one vertex at a time
                adj = best_adj[:]
                for new_v in range(best_n, target_n):
                    adj.append(0)
                    # Connect new vertex to existing ones with random probability
                    for i in range(new_v):
                        if rng.random() < 0.5:
                            adj[new_v] |= (1 << i)
                            adj[i] |= (1 << new_v)
                if _valid(adj, target_n):
                    found = True
                    best_n = target_n
                    best_adj = adj
                    best_edges = _edges_from_adj(adj, target_n)
                    break
                else:
                    improved = _try_local_search(adj, target_n, rng, max_iters=500)
                    if _valid(improved, target_n):
                        found = True
                        best_n = target_n
                        best_adj = improved
                        best_edges = _edges_from_adj(improved, target_n)
                        break
        
        if found:
            continue
        
        # Could not find valid graph for this target_n; stop trying higher
        break

    # Final validation
    if _valid(best_adj, best_n):
        return {'n': best_n, 'edges': best_edges}
    
    # Fallback
    return {'n': 5, 'edges': []}
