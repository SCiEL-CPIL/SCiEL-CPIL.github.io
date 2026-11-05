import time
import random
import itertools
import numpy as np

_START = time.time()
_DEADLINE = _START + 115.0  # safety margin before 150s kill

def _deadline():
    return time.time() >= _DEADLINE

def _adj_to_numpy(adj, n):
    """Convert adjacency list (bitmasks) to numpy adjacency matrix."""
    mat = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        mask = adj[i]
        for j in range(i + 1, n):
            if mask & (1 << j):
                mat[i, j] = 1
                mat[j, i] = 1
    return mat

def _numpy_has_regular_subgraph(mat, n, min_size=6):
    """Vectorized check: does any subset of size >= min_size induce a regular subgraph?"""
    # Precompute all masks as indices
    vertices = list(range(n))
    for k in range(min_size, n + 1):
        if _deadline():
            return True  # assume invalid to be safe
        for comb in itertools.combinations(vertices, k):
            idx = np.array(comb, dtype=np.intp)
            sub = mat[np.ix_(idx, idx)]
            degs = sub.sum(axis=1)
            if len(set(degs.tolist())) == 1:
                return True
    return False

def _numpy_count_regular_subgraphs(mat, n, min_size=6):
    """Count regular induced subgraphs of size >= min_size using numpy."""
    cnt = 0
    vertices = list(range(n))
    for k in range(min_size, n + 1):
        if _deadline():
            return cnt
        for comb in itertools.combinations(vertices, k):
            idx = np.array(comb, dtype=np.intp)
            sub = mat[np.ix_(idx, idx)]
            degs = sub.sum(axis=1)
            if len(set(degs.tolist())) == 1:
                cnt += 1
    return cnt

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

def _try_local_search(adj, n, rng, max_iters=2000):
    """Local search: flip single edges to eliminate regular subgraphs."""
    if _deadline():
        return adj
    best = adj[:]
    best_cnt = _count_regular_subgraphs(best, n, 6)
    if best_cnt == 0:
        return best
    
    cur = adj[:]
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
        if no_improve > 200:
            # Reset with random perturbation
            for _ in range(5):
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

def _construct_comparability_graph(n, rng):
    """
    Construct a graph based on a random total order (poset) where edges connect
    pairs with distance greater than a threshold. This structurally enforces
    degree heterogeneity in large induced subgraphs.
    """
    adj = [0] * n
    perm = list(range(n))
    rng.shuffle(perm)
    pos = [0] * n
    for i, v in enumerate(perm):
        pos[v] = i
    
    # Use multiple thresholds to create varied structure
    threshold1 = rng.uniform(0.2, 0.4) * n
    threshold2 = rng.uniform(0.5, 0.7) * n
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = abs(pos[i] - pos[j])
            # Edge if distance is in a "sweet spot" - not too small, not too large
            in_band1 = threshold1 <= dist <= threshold2
            # Add some randomness
            if in_band1 and rng.random() < 0.8:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
            elif not in_band1 and rng.random() < 0.15:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _construct_tournament_based(n, rng):
    """
    Use a random tournament structure to break regularity.
    For each pair, orient the edge based on a random ordering, then
    add the edge with probability depending on the orientation strength.
    """
    adj = [0] * n
    # Random scores for each vertex
    scores = [rng.random() for _ in range(n)]
    order = sorted(range(n), key=lambda v: scores[v])
    rank = [0] * n
    for i, v in enumerate(order):
        rank[v] = i
    
    for i in range(n):
        for j in range(i + 1, n):
            # Probability based on rank difference
            rank_diff = abs(rank[i] - rank[j])
            p = 0.3 + 0.4 * (rank_diff / max(1, n - 1))
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
        
        # Strategy 1: Comparability graph constructions (new)
        for attempt in range(80):
            if _deadline():
                break
            adj = _construct_comparability_graph(target_n, rng)
            if _valid(adj, target_n):
                found = True
                best_n = target_n
                best_adj = adj
                best_edges = _edges_from_adj(adj, target_n)
                break
            else:
                improved = _try_local_search(adj, target_n, rng, max_iters=1500)
                if _valid(improved, target_n):
                    found = True
                    best_n = target_n
                    best_adj = improved
                    best_edges = _edges_from_adj(improved, target_n)
                    break
        
        if found:
            continue
        
        # Strategy 2: Tournament-based constructions (new)
        for attempt in range(80):
            if _deadline():
                break
            adj = _construct_tournament_based(target_n, rng)
            if _valid(adj, target_n):
                found = True
                best_n = target_n
                best_adj = adj
                best_edges = _edges_from_adj(adj, target_n)
                break
            else:
                improved = _try_local_search(adj, target_n, rng, max_iters=1500)
                if _valid(improved, target_n):
                    found = True
                    best_n = target_n
                    best_adj = improved
                    best_edges = _edges_from_adj(improved, target_n)
                    break
        
        if found:
            continue
        
        # Strategy 3: Random graphs with various densities
        for attempt in range(60):
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
                improved = _try_local_search(adj, target_n, rng, max_iters=1500)
                if _valid(improved, target_n):
                    found = True
                    best_n = target_n
                    best_adj = improved
                    best_edges = _edges_from_adj(improved, target_n)
                    break
        
        if found:
            continue
        
        # Strategy 4: Start from a smaller valid graph and try to extend
        if best_n >= 6 and best_n < target_n:
            for attempt in range(30):
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
                    improved = _try_local_search(adj, target_n, rng, max_iters=1000)
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
